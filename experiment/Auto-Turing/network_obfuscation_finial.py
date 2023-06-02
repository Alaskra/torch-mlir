import torch_mlir, torch
from models import LeNet
import random
import time
import os
import re
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
import argparse
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 通用Pass列表
general_obfuscation = {
    "a": "torch-insert-skip{layer=1}",
    "e": "torch-value-split{number=2 layer=1}",
    "f": "torch-mask-split{number=2 layer=1}",
    "i": "torch-insert-RNNWithZeros{activationFunc=tanh number=1 layer=1}"
}

# 激活函数列表
activationFunc = ["tanh", "sigmoid"]

# 定义问题
obfuscation_name = [
    ["a", "layer=1"], ["e", "number=2", "layer=1"], ["f", "number=2", "layer=1"], ["i", "activationFunc=tanh", "number=1", "layer=1"]
]
obfuscation_name_layer = ["a"]
obfuscation_name_num_layer = ["e", "f"]
obfuscation_name_act_num_layer = ["i"]

# 初始化网络
net_origin = LeNet()
net_obfu = LeNet()
module = torch_mlir.compile(
        net_origin, torch.ones(1, 1, 28, 28), output_type="torch", use_tracing=True, ignore_traced_shapes=True
    )

# 保存目录
directory = "/root/mycode/torch-mlir/experiment/Auto-Turing"

# 获取神经网络的MLIR代码长度
def get_length(module_l):
    file_path = os.path.join(directory, "origin_model.txt")
    output = module_l.operation.get_asm(large_elements_limit=10)
    with open(file_path, "w") as file:
        file.write(output)
    
    # 输出参数个数
    content = "return"
    with open(file_path, "r") as file:
         for line in file:
            if content in line:
                line_result = line.strip()
    match = re.search(r'%(\d+)', line_result)
    number = match.group(1)
    return number

# 计算混淆前后的时间差值
def get_time_origin():
    start_time_origin = time.time()
    module_origin = torch_mlir.compile(net_origin, torch.ones(1, 1, 28, 28), output_type="linalg-on-tensors")

    backend_origin = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled_origin = backend_origin.compile(module_origin)
    jit_module_origin = backend_origin.load(compiled_origin)
    jit_func_origin = jit_module_origin.forward
    output_origin = jit_func_origin(torch.ones(1, 1, 28, 28).numpy())
    end_time_origin = time.time()
    return (end_time_origin - start_time_origin)

time_origin = get_time_origin()

# 初始化种群
def create_individual(chromosome_length):
    return [random.choice(obfuscation_name) for _ in range(chromosome_length)]

# 定义均匀交叉算法
def uniform_crossover(parent1, parent2, chromosome_length):
    child1 = []
    child2 = []

    for i in range(chromosome_length):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    return child1, child2

# 锦标赛选择算法
def tournament_selection(population, fitness_values, tournament_size):
    selected_population = []

    for _ in range(len(population)):
        participants = random.sample(range(len(population)), tournament_size)
        winner = None
        max_fitness = float('-inf')

        for participant in participants:
            if fitness_values[participant] > max_fitness:
                max_fitness = fitness_values[participant]
                winner = participant

        selected_population.append(population[winner])

    return selected_population
    
# 定义目标函数
def fitness(individual, time_origin, origin_length):
    name_obfu = []
    pass_obfu = []
    for name in individual:
        name_obfu.append(name[0])
        passes = general_obfuscation.get(name[0])
        if (name[0] in obfuscation_name_layer):
            pattern_layer = r"layer=\d+"
            match_layer = re.search(pattern_layer, passes)
            layer_origin = match_layer.group()
            passes = passes.replace(layer_origin, name[1])
        elif (name[0] in obfuscation_name_num_layer):
            pattern_layer = r"layer=\d+"
            pattern_num = r"number=\d+"
            match_num = re.search(pattern_num, passes)
            num_origin = match_num.group()
            passes = passes.replace(num_origin, name[1])
            match_layer = re.search(pattern_layer, passes)
            layer_origin = match_layer.group()
            passes = passes.replace(layer_origin, name[2])
        elif (name[0] in obfuscation_name_act_num_layer):
            pattern_act = r"activationFunc=\w+"
            pattern_layer = r"layer=\d+"
            pattern_num = r"number=\d+"
            match_act = re.search(pattern_act, passes)
            act_origin = match_act.group()
            passes = passes.replace(act_origin, name[1])
            match_num = re.search(pattern_num, passes)
            num_origin = match_num.group()
            passes = passes.replace(num_origin, name[2])
            match_layer = re.search(pattern_layer, passes)
            layer_origin = match_layer.group()
            passes = passes.replace(layer_origin, name[3])
        pass_obfu.append(passes)
            
    start_time_obfu = time.time()
    module_obfu = torch_mlir.compile(net_obfu, torch.ones(1, 1, 28, 28), output_type="torch")
    for i in range(0, len(name_obfu) - 1):
        logging.debug("Performing obfuscation and compile, name: %s, pass: %s", name_obfu[i], pass_obfu[i])
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module_obfu,
        f"builtin.module(func.func({pass_obfu[i]}))",
        name_obfu[i],
    )
    obfu_length = int(get_length(module_obfu))
    logging.debug("Performing Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR, name: %s, pass: %s", name_obfu, pass_obfu)
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module_obfu,
    "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
    "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
    )
    backend_obfu = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled_obfu = backend_obfu.compile(module_obfu)
    jit_module_obfu = backend_obfu.load(compiled_obfu)
    jit_func_obfu = jit_module_obfu.forward
    output_obfu = jit_func_obfu(torch.ones(1, 1, 28, 28).numpy())
    end_time_obfu = time.time()
    time_obfu = end_time_obfu - start_time_obfu
    return round(((obfu_length - origin_length) / origin_length) / ((time_obfu - time_origin)/ time_origin + 1)**2, 4)

# 遗传算法主循环
def genetic_algorithm(max_genetarion, crossover_rate, mutation_rate_1, mutation_rate_2, 
                      population_size, chromosome_length, tournament_size):
    # 计算混淆前的神经网络的mlir的代码长度
    origin_length = int(get_length(module))

    # 计算混淆前的神经网络的运行时间
    time_origin = get_time_origin()

    # 初始化种群
    population = [create_individual(chromosome_length) for i in range(0, population_size)]

    for generation in range(max_genetarion):
        # 计算适应度值
        fitness_values = [fitness(individual, time_origin, origin_length) for individual in population]
        
        # 选择操作
        logging.info(f"Generation: {generation}")
        selected_population = tournament_selection(population, fitness_values, tournament_size)

        # 交叉操作
        logging.debug("Performing crossover...")
        next_generation = []
        for i in range(population_size):
            parent1 = selected_population[i]
            if random.random() < crossover_rate:
                parent2 = random.choice(selected_population)
                child1, child2 = uniform_crossover(parent1, parent2, chromosome_length)
                next_generation.append(child1)
                next_generation.append(child2)
            else:
                next_generation.append(parent1)
        
        # 变异操作
        logging.debug("Performing mutation...")
        for i in range(population_size):
            individual = next_generation[i]
            for j in range(chromosome_length):
                if random.random() < mutation_rate_1:
                    individual[j] = random.choice(list(obfuscation_name))
                if random.random() < mutation_rate_2:
                    if (individual[j][0] in obfuscation_name_layer):
                        layer_random = 1
                        individual[j][1] = f"layer={layer_random}"
                    elif(individual[j][0] in obfuscation_name_num_layer):
                        layer_random = random.randint(1, 2)
                        num_random = random.randint(2, 4)
                        individual[j][1] = f"number={num_random}"
                        individual[j][2] = f"layer={layer_random}"
                    elif(individual[j][0] in obfuscation_name_act_num_layer):
                        act_random = random.choice(activationFunc)
                        layer_random = random.randint(1, 2)
                        num_random = random.randint(2, 6)
                        individual[j][1] = f"activationFunc={act_random}"
                        individual[j][2] = f"number={num_random}"
                        individual[j][3] = f"layer={layer_random}"
        
        # 更新种群
        population = next_generation

    # 输出结果
    best_individual = max(population, key=lambda x: fitness(x, time_origin, origin_length))
    file_path = os.path.join(directory, "output.txt")
    best_fitness = fitness(best_individual, time_origin, origin_length)
    with open(file_path, "a+") as file:
        file.write("Best solution: %s\n" % best_individual)
        file.write("Best fitness_values: %s\n" % best_fitness)
    # logging.info("Best solution: %s", best_individual)
    # logging.info("Best fitness_values: %s", fitness(best_individual, time_origin, origin_length))

def _get_argparse():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm")
    parser.add_argument(
        "-m",
        "--max_genetarion",
        type=int,
        default=30,
        help="""Specifies the maximum number of iterations for the genetic algorithm"""
    )
    parser.add_argument(
        "-c",
        "--crossover_rate",
        type=float,
        default=0.8,
        help="""Specifies the crossover probability for the genetic algorithm"""
    )
    parser.add_argument(
        "-m1",
        "--mutation_rate_1",
        type=float,
        default=0.2,
        help="""Specifies the first mutation probability of the genetic algorithm"""
    )
    parser.add_argument(
        "-m2",
        "--mutation_rate_2",
        type=float,
        default=0.5,
        help="""Specifies the second mutation probability for the genetic algorithm"""
    )
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        default=20,
        help="""Specifies the population size for the genetic algorithm"""
    )
    parser.add_argument(
        "-cl",
        "--chromosome_length",
        type=int,
        default=5,
        help="""Specifies the chromosome length for the genetic algorithm"""
    )
    parser.add_argument(
        "-t",
        "--tournament_size",
        type=int,
        default=3,
        help="""Specifies the tournament size for the genetic algorithm"""
    )
    parser.add_argument(
        "-b",
        "--budget",
        type=float,
        default=0.2,
        help="""Specifies the delay budget for the genetic algorithm"""
    )
    return parser

if __name__ == "__main__":
    args = _get_argparse().parse_args()
    for i in range(0, 5):
        genetic_algorithm(args.max_genetarion, args.crossover_rate, args.mutation_rate_1, 
                        args.mutation_rate_2, args.population_size, args.chromosome_length, 
                        args.tournament_size)
    