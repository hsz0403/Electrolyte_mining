import openai
import requests
from openai import OpenAI
import csv
import PyPDF2
import re
import os
import requests
import pandas as pd
import tiktoken
import time
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
import ast

def chat(context):
    attempts = 3
    response_msgs=[]
    while attempts > 0:
        try:
            
            messages_multi_turn=[
                    {"role": "system", "content": "I'm working on projects for text mining using LLM, please give me help when I need."},
                    
                ]

            while True:       # 获取用户输入       
                user_input = input("You: ")    
                if user_input.lower() == 'quit':  # 如果用户输入\'quit\'，则退出循环           
                    break   
                messages_multi_turn.append({"role": "user", "content": user_input})  
                response = client.chat.completions.create(
                    model='gpt-4-turbo-preview',messages=messages_multi_turn
                    
                )
                bot_reply = response.choices[0].message.content
                print("Bot:", bot_reply)                      # 将bot的回复也加入到消息历史中，以便进行下 一轮的对话           
                messages_multi_turn.append({"role": "assistant", "content": bot_reply})       
    
                answers = response.choices[0].message.content
            break

        except Exception as e:
            attempts -= 1
            if attempts > 0:
                print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 2)")
                time.sleep(60)
            else:
                print(f"Error: Failed to process. Skipping. (model 2)")
                answers = "No"
                break

    response_msgs.append(answers)
    return response_msgs


'''messages = [ {"role": "system", "content": "I\'m working on projects for text mining using LLM, please give me help when I need."} ]  
print("Chatbot is ready. Start chatting with it! Type \'quit\' to exit.")     
while True:       # 获取用户输入       
    user_input = input("You: ")       
    if user_input.lower() == 'quit':  # 如果用户输入\'quit\'，则退出循环           
        break              messages.append({"role": "user", "content": user_input})              try:           # 调用OpenAI的API进行聊天回复           
    response = openai.ChatCompletion.create(             model='gpt-4-turbo-preview',  messages=messages           )          
    bot_reply = response.choices[0].message['content']           
    print("Bot:", bot_reply)                      # 将bot的回复也加入到消息历史中，以便进行下 一轮的对话           
    messages.append({"role": "assistant", "content": bot_reply})       
    except Exception as e:           
        print(f"An error occurred: {str(e)}")\n\nif __name__ == "__main__":   multi_turn_chat()\n
'''
if __name__=='__main__': #e.g. openai.api_key = "abcdefg123abc" 
    


    gpt4='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'
    #claude='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'

    client = OpenAI(
    api_key=gpt4,  # this is also the default, it can be omitted
    )
    
    code_scratch="""def chat(context):
    attempts = 3
    response_msgs=[]
    while attempts > 0:
        try:
            
            

            
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                messages=[
                    {"role": "system", "content": "I'm working on projects for text mining using LLM, please give me help when I need."},
                    {"role": "user", "content": context}
                ]
            )
            answers = response.choices[0].message.content
            break

        except Exception as e:
            attempts -= 1
            if attempts > 0:
                print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 2)")
                time.sleep(60)
            else:
                print(f"Error: Failed to process. Skipping. (model 2)")
                answers = "No"
                break

    response_msgs.append(answers)
    return response_msgs"""
    
    message=chat("")
    #print(message)