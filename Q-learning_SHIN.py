# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:57:44 2022

@author: parkh
"""

import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.s= ""
        
    def step(self, a):
        if a==0:
            if self.s == "01010":
                reward = +1000
            else:
                reward = -1
            self.move_left()
        elif a==1:
            self.move_right()
            reward = +1    
        done = self.is_done()
        return self.s, reward, done
    
    def move_left(self):
        self.s = self.s+"0"
        
    def move_right(self):
        self.s = self.s+"1"
    
    def is_done(self):
        if len(self.s) == 6:
            return True
        else:
            return False
        
    def reset(self):
        self.s = ""
        return self.s
    
class QAgent():
    def __init__(self):
        self.q_table = np.zeros((127,2))
        self.eps = 0.9
        self.alpha = 0.1
    
    def get_fitness(self, s):
        fitness = 0
        for i in range(len(s)):
            if s[-i-1]=="0":
                fitness = fitness + 2**(i)
            else:
                fitness = fitness + 2 * 2**(i)
        return fitness
        
    def select_action(self, s):
        coin = random.random()
        k = self.get_fitness(s)
        if coin < self.eps:
            action = random.randint(0,1)
        else:
            action_val = self.q_table[k,:]
            action = np.argmax(action_val)
        return action
    
    def select_action2(self, s):
        k = self.get_fitness(s)
        action_val = self.q_table[k,:]
        action = np.argmax(action_val)
        return action
    
    def update_table(self, transition):
        #"",1,1,1,False
        s, a, r, s_prime = transition
        k = self.get_fitness(s)
        #k=0
        next_k = s_prime
        next_k=self.get_fitness(next_k)
        #SARSA 업데이트 식을 이용
        self.q_table[k,a] = self.q_table[k,a] + self.alpha * (r + np.amax(self.q_table[next_k, :]) - self.q_table[k,a])
            
    def anneal_eps(self):
        self.eps -=0.001
        self.eps = max(self.eps, 0.2)
        
    def show_table(self):
        q_lst = self.q_table.tolist()
        data = np.zeros((127,2))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
    
    def show(self):
        print(self.q_table.tolist())
        
def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000):
        done = False
        
        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s = s_prime
        agent.anneal_eps()
    #agent.show()
    
    done=False
    s=env.reset()
    r_sum=0
    while not done:
        a = agent.select_action2(s)
        s_prime, r, done = env.step(a)
        r_sum= r_sum+r
        s = s_prime
    #print(s,r_sum)
    #agent.show_table()
    return r_sum
#main()
av=0
for i in range(100):
    r_sum=main()
    print(i+1 , "회 최적정책 리워드는 ", r_sum)
    av=r_sum+av
print(av/100,"은 평균")