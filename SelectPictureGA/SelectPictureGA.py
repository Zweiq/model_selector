# !/bin/python
# -*- coding: utf-8 -*-

"""
# 利用GA挑选合适图片
"""

import matplotlib.pyplot as plt
from deap import creator
from deap import tools
from deap import base
from PIL import Image
import random
import json

def compound_picture(backgound_path, label_path, theme_path):
    # 加载背景图片
    bg_img = Image.open(backgound_path).convert("RGBA")

    # 定义标签位置和主题位置
    label_box = (0, 0, 100, 100)
    theme_box = (100, 100, 300, 300)

    # 加载并且重置背景和主题图片
    label_img = Image.open(label_path).resize((100, 100))
    theme_img = Image.open(theme_path).resize((200, 200))

    bg_img.paste(label_img, label_box, label_img)
    bg_img.paste(theme_img, theme_box, theme_img)

    return bg_img

def WriteRandom(data_list):
    output_list = {}
    for i in data_list:
        value = random.randint(1, 100)
        output_list[str(i)]=value
    with open("score.json", "w", encoding="utf-8") as f:
        json.dump(output_list, f)


def select_parameter(hyper_para):
    hyper_list = []
    for i in hyper_para:
        hyper_list.append(random.choice(i))
    return creator.Individual(hyper_list)

def eva_max(individual):
    with open("score.json", "r") as f:
        score = json.load(f)
    return float(score[str(individual)]),

def mutCounter(individual, para_grid):
    if random.random()>0.5:
        index = random.randint(0, len(para_grid)-1)
        individual[index] = random.choice(para_grid[index])
    else:
        pass
    return individual,

def SelectPictureGA(para_grid):
    """
    利用GA进行素材调优
    """
    # 初始化
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 初始种群生成
    toolbox = base.Toolbox()
    toolbox.register("attr_float", select_parameter, para_grid)
    toolbox.register("population", tools.initRepeat, list, toolbox.attr_float)

    # 进化器生成
    toolbox.register("evaluate", eva_max)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutCounter, para_grid=para_grid)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 生成器初始化参数
    population = toolbox.population(n=40)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 50

    # 此处添加模拟点击率函数
    WriteRandom(population)
    # 模拟点击率函数结束

    # 选择初始化种群
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # 此处添加模拟点击率函数
        WriteRandom(invalid_ind)
        # 模拟点击率函数结束
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        population[:] = offspring
    return population

if __name__ == '__main__':
    picture_data = [['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'],
                    ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7'],
                    ['t1', 't2', 't3', 't4', 't5', 't6', 't7']]
    best_data = SelectPictureGA(picture_data)[0:4]
    for i in range(len(best_data)):
        img = compound_picture(backgound_path="./picture/background/"+best_data[i][0]+".png",
                               label_path="./picture/label/"+best_data[i][1]+".png",
                               theme_path="./picture/theme/"+best_data[i][2]+".png")
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()



