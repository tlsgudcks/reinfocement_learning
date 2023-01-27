# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 00:17:44 2022

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
            reward = +1    
            self.move_right()    
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
        self.q_table = np.zeros((2048,2))
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
        s, a, r, s_prime = transition
        k = self.get_fitness(s)
        next_k = s_prime
        a_prime = self.select_action(s_prime) 
        next_k=self.get_fitness(next_k)
        self.q_table[k,a] = self.q_table[k,a] + self.alpha * (r + self.q_table[next_k, a_prime] - self.q_table[k,a])
            
    def anneal_eps(self):
        self.eps -=0.001
        self.eps = max(self.eps, 0.2)
    
    def show(self):
        print(self.q_table.tolist())
        print(self.eps)
        
        
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
        #if n_epi%10==0 or n_epi<10:
        #    print(n_epi,"에피소드가 지남")
        #    agent.show()
    
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