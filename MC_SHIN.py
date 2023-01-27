# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:42:32 2022

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
        else:
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
        self.q_table = np.zeros((100000000,6))
        self.eps = 0.9
        self.alpha = 0.01
    
    def get_state(self, s):
        state = 0
        for i in range(len(s)):
            if s[-i-1]=="0":
                state = state + 2**(i)
            else:
                state = state + 2 * 2**(i)
        return state
        
    def select_action(self, s):
        coin = random.random()
        k = self.get_state(s)
        if coin < self.eps:
            action = random.randint(0,1)
        else:
            action_val = self.q_table[k,:]
            action = np.argmax(action_val)
        return action
    
    def select_action2(self, s):
        k = self.get_state(s)
        action_val = self.q_table[k,:]
        action = np.argmax(action_val)
        return action
    
    def update_table(self, history):
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            k = self.get_state(s)
            cum_reward = cum_reward+r
            self.q_table[k,a] = self.q_table[k,a] + self.alpha * (cum_reward - self.q_table[k,a])
            
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
        history=[]
        
        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s,a,r,s_prime))
            s = s_prime
        agent.update_table(history)
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
    #agent.show()
    
    return r_sum 
    #agent.show_table()
av=0
for i in range(100):
    r_sum=main()
    print(i+1 , "회 최적정책 리워드는 ", r_sum)
    av=r_sum+av
print(av/100,"은 평균")