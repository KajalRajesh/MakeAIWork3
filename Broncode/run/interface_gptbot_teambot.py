

#! /home/pans/miniconda3/envs/miw/bin/python

import os
import time
import torch
import openai
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from sentence_transformers import SentenceTransformer
# Imports for Interface
import tkinter
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import ttk, messagebox

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants and hyperparameters
num_classes = 4 # Can be changed to 2 classes if you want to check on good/bad apples
image_size = 224
batch_size = 64
apples_quantity = 300

# Generate test dir, classes and dataset by number of classes and device
if num_classes == 2:
    if device == torch.device('cuda'):
        test_dir_resnet = "./../../models/ResNet_GPU_Class_2.pth"
        test_dir_team = "./../../models/TeamModel_GPU_Class_2.pth"
    else:
        test_dir_resnet = "./../../models/ResNet_CPU_Class_2.pth"
        test_dir_team = "./../../models/TeamModel_CPU_Class_2.pth"
    test_dir_data = "./../../data/Test_class_2"
    classes = ('0', '1')
else:
    if device == torch.device('cuda'):
        test_dir_resnet = "./../../models/ResNet_GPU_Class_4.pth"
        test_dir_team = "./../../models/TeamModel_GPU_Class_4.pth"
    else:
        test_dir_resnet = "./../../models/ResNet_CPU_Class_4.pth"
        test_dir_team = "./../../models/TeamModel_CPU_Class_4.pth"
    test_dir_data = "./../../data/Test_class_4"
    classes = ('0', '1', '2', '3')
    
# Load the pre-trained ResNet model
model_resnet = torch.hub.load("pytorch/vision", "resnet18" , weights="IMAGENET1K_V1")
num_features = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_features, num_classes)
model_resnet = model_resnet.to(device)
model_resnet.load_state_dict(torch.load(test_dir_resnet))
model_resnet.eval()

# Load the pre-trained Team model
model_team = torch.load(test_dir_team)
model_team = model_team.to(device)
model_team.eval()

# Define the transformations for test data
transform_test = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the test dataset
test_dir = test_dir_data
test_dataset = ImageFolder(test_dir, transform=transform_test)

# Reduce the test dataset to a random sample of apples_quantity examples
test_dataset = random.sample(list(test_dataset), k=apples_quantity)

# Create a DataLoader for managing the test data batches
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function for predicted apples with models
def get_model(data_loader, model):
    y_pred = []
    y_true = []
    # Iterate over the test data and make predictions for ResNet and Team model
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        output = torch.argmax(torch.exp(output), dim=1).cpu().numpy()
        y_pred.extend(output)
        labels = labels.cpu().numpy()
        y_true.extend(labels)
    
    return y_pred, y_true

# Create test loader for ResNet and Team model
y_pred_resnet, y_true_resnet = get_model(test_loader, model_resnet)
y_pred_team, y_true_team = get_model(test_loader, model_team)


# Create empty lists to store the predicted labels and true labels
predicted_labels_resnet = []
true_labels_resnet = []

predicted_labels_team = []
true_labels_team = []

test_loss_resnet = 0.0
test_loss_team = 0.0
correct_resnet = 0
correct_team = 0

criterion = nn.CrossEntropyLoss()

# Disable gradient calculation
with torch.no_grad():
    # Set the models to evaluation mode
    model_resnet.eval()
    model_team.eval()
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass for resnet18 model
        outputs_resnet = model_resnet(images)
        _, predicted_resnet = torch.max(outputs_resnet, 1)
        
        # Calculate loss for resnet18 model
        loss_resnet = criterion(outputs_resnet, labels)
        test_loss_resnet += loss_resnet.item()
        
        # Update the counts for correct predictions for resnet18 model
        correct_resnet += (predicted_resnet == labels).sum().item()
        
        # Append predicted and true labels for resnet18 model
        predicted_labels_resnet.extend(predicted_resnet.cpu().numpy())
        true_labels_resnet.extend(labels.cpu().numpy())
        
        # Forward pass for team model
        outputs_team = model_team(images)
        _, predicted_team = torch.max(outputs_team, 1)
        
        # Calculate loss for team model
        loss_team = criterion(outputs_team, labels)
        test_loss_team += loss_team.item()
        
        # Update the counts for correct predictions for team model
        correct_team += (predicted_team == labels).sum().item()
        
        # Append predicted and true labels for team model
        predicted_labels_team.extend(predicted_team.cpu().numpy())
        true_labels_team.extend(labels.cpu().numpy())

# Calculate the average test losses
test_loss_resnet /= len(test_loader)
test_loss_team   /= len(test_loader)

# Calculate the test accuracies
accuracy_resnet = correct_resnet / len(test_dataset)
accuracy_team   = correct_team / len(test_dataset)

# Make resnet and Team model test accuracy and test loss
resnet_test_accuracy = "{:.2f}%".format(accuracy_resnet * 100)
resnet_test_loss     = "{:.4f}".format(test_loss_resnet)
team_test_accuracy   = "{:.2f}%".format(accuracy_team * 100)
team_test_loss       = "{:.4f}".format(test_loss_team)


# Function for error message box
def show_error(error_message):
    messagebox.showerror('Input Error', f'Error: {error_message}')
    
# Function for counting apples in classes    
def calculate_bad_apple_percentage(sample, num_classes):
    if num_classes == 2:
        blotch_apples = 0
        normal_apples = sample.count(1)
        rotten_apples = sample.count(2)
        scab_apples   = 0
        return ((rotten_apples + blotch_apples + scab_apples) / len(sample)) * 100
    else:
        blotch_apples = sample.count(0)
        normal_apples = sample.count(1)
        rotten_apples = sample.count(2)
        scab_apples   = sample.count(3)
        return ((rotten_apples + blotch_apples + scab_apples) / len(sample)) * 100
    
# Function for genarate AQL
def classify_batch(perc_bad_apples, aql):
    if perc_bad_apples <= aql:
        return 'Class 1: Supermarket'
    elif aql < perc_bad_apples < 6.5:
        return 'Class 2: Apple sauce factory'
    elif 6.5 <= perc_bad_apples < 15:
        return 'Class 3: Apple syrup factory'
    else:
        return 'Class 4: Feed them to the pigs!'
    
# Function to label the apple classes  
def labeled_apple(number_classes):
    if number_classes == 2:
        labels = ['Normal', 'Rotten']
    else:
        labels = ['Blotch', 'Normal', 'Rotten', 'Scab']
        return labels

# Function to create a pie chart
def generate_pie_chart(sizes, labels, title, filename):
    plt.clf()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(title)
    plt.savefig(filename)

# Function to generate models
def calculate():
    try:
        # Get input from user
        batch_size = int(batch_size_2.get())
        sample_size = int(sample_size_2.get())
        number_of_runs = int(nr_runs.get())
        aql = float(user_aql.get())

        # Make limit for inputs
        if not 10 <= batch_size <= len(y_pred_resnet):
            show_error(f'Batch size must be between 10 and {len(y_pred_resnet)}')
            return
        if not 5 <= sample_size <= 100:
            show_error('Sample size must be between 5 and 100')
            return
        if not 1 <= number_of_runs <= 100:
            show_error('Number of runs must be between 1 and 50')
            return
        if not 0 < aql < 100:
            show_error('Your AQL must be between 0 and 100')
            return
        if sample_size * number_of_runs > batch_size:
            show_error("The multiplication of 'sample size' and 'runs number' must be less than 'batch size'")
            return

        # Generate random indices for batch size and sample size
        random.seed(f'{time.time()/200:.1f}')
        random_numbers_batch = random.sample(range(len(y_pred_resnet)), batch_size)
        random_numbers_sample = random.sample(range(len(random_numbers_batch)), (sample_size * number_of_runs))

        # Create batch size with batch index and y_pred
        batch_size_resnet = [y_pred_resnet[index] for index in random_numbers_batch]
        batch_size_team = [y_pred_team[index] for index in random_numbers_batch]

        # Create sample size with sample index and batch size
        sample_resnet = [batch_size_resnet[index] for index in random_numbers_sample]
        sample_team = [batch_size_team[index] for index in random_numbers_sample]

        # Calculate the percentage of bad apples for ResNet model
        perc_bad_apples_resnet = calculate_bad_apple_percentage(sample_resnet, num_classes)

        # Classify the batch based on the percentage of bad apples for ResNet model
        answer_resnet = classify_batch(perc_bad_apples_resnet, aql)

        # Calculate the percentage of bad apples for Team model
        perc_bad_apples_team = calculate_bad_apple_percentage(sample_team, num_classes)

        # Classify the batch based on the percentage of bad apples for Team model
        answer_team = classify_batch(perc_bad_apples_team, aql)
        
        # Labeled apples for both model
        label = labeled_apple(num_classes)

        # Generate pie chart for ResNet model
        sizes_resnet = [sample_resnet.count(i) for i in range(num_classes)]
        generate_pie_chart(sizes_resnet, label, 'Distribution of Apple Types by ResNet18 in samples', './../../data/charts/Pie_chart_ResNet.png')

        # Generate pie chart for Team model
        sizes_team = [sample_team.count(i) for i in range(num_classes)] 
        generate_pie_chart(sizes_team, label, 'Distribution of Apple Types by Team model in samples', './../../data/charts/Pie_chart_Team.png')

        # Update interface with results of ResNet model
        good_apple_percentage_resnet.set(f"{(100 - perc_bad_apples_resnet):.2f} %")
        test_accuracy_resnet.set(f"{resnet_test_accuracy} %")
        test_loss_resnet.set(f"{resnet_test_loss} %")
        group_apple_category_resnet.set(answer_resnet)

        # Update interface with results of Team model
        good_apple_percentage_team.set(f"{(100 - perc_bad_apples_team):.2f} %")
        test_accuracy_team.set(f"{team_test_accuracy} %")
        test_loss_team.set(f"{team_test_loss} %")
        group_apple_category_team.set(answer_team)

    except ValueError:
        pass

# Function to connect model to chatGPT
def gpt_chatbot():
    try:
        # Get input from user
        batch_size = int(batch_size_2.get())
        sample_size = int(sample_size_2.get())
        number_of_runs = int(nr_runs.get())
        aql = float(user_aql.get())
        openai.api_key = f'{open_ai_access_key.get()}'
        
        # Set variables for chat bot
        name = "Make IT Work ResNet bot"
        bot_name = "Make IT Work ResNet bot"
        good = 'Normal'
        bad  = 'Rot + Blotch + Scab'
        sauce_factory_aql = 6.5
        syrup_factory_aql = 15
        
        # Generate random indices for batch size and sample size
        random.seed(f'{time.time()/200:.1f}')
        random_numbers_batch  = random.sample(range(len(y_pred_resnet)), batch_size)
        random_numbers_sample = random.sample(range(len(random_numbers_batch)), (sample_size * number_of_runs))

        # Create batch size with batch index and y_pred
        batch_size_resnet = [y_pred_resnet[index] for index in random_numbers_batch]

        # Create sample size with sample index and batch size
        samples_resnet = [batch_size_resnet[index] for index in random_numbers_sample]
        
        # Calculate the percentage of bad apples with ResNet model
        if num_classes == 2:
            # For 2 classes
            Blotch_Apple_resnet_batch = 0
            Normal_Apple_resnet_batch = batch_size_resnet.count(0)
            Rot_Apple_resnet_batch    = batch_size_resnet.count(1)
            Scab_Apple_resnet_batch   = 0
            Blotch_Apple_resnet_samples  = 0
            Normal_Apple_resnet_samples  = samples_resnet.count(0)
            Rot_Apple_resnet_samples     = samples_resnet.count(1)
            Scab_Apple_resnet_samples    = 0
            
        else:
            # For 4 classes
            Blotch_Apple_resnet_batch = batch_size_resnet.count(0)
            Normal_Apple_resnet_batch = batch_size_resnet.count(1)
            Rot_Apple_resnet_batch    = batch_size_resnet.count(2)
            Scab_Apple_resnet_batch   = batch_size_resnet.count(3)
            Blotch_Apple_resnet_samples  = samples_resnet.count(0)
            Normal_Apple_resnet_samples  = samples_resnet.count(1)
            Rot_Apple_resnet_samples     = samples_resnet.count(2)
            Scab_Apple_resnet_samples    = samples_resnet.count(3)
             
        number_of_bad_apples_resnet_batch   = (Rot_Apple_resnet_batch + Blotch_Apple_resnet_batch + Scab_Apple_resnet_batch)
        number_of_bad_apples_resnet_samples = (Rot_Apple_resnet_samples + Blotch_Apple_resnet_samples + Scab_Apple_resnet_samples)
 
        # ChatGPT function
        def chat_with_gpt3(prompt):
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                n=1,
                stop=None
            )
            if len(response.choices) > 0:
                return response.choices[0].text.strip()
            return ""
        
        # Make initial information for chat bot
        initial_prompt = f"Name: {name}\nBatch size: {batch_size}\nSamples size: {sample_size*number_of_runs}\nSupermarket Aql: {aql}\nsyrup factory Aql: {syrup_factory_aql}\nsauce factory Aql: {sauce_factory_aql}\nBlotch in batch: {Blotch_Apple_resnet_batch}\nScab in batch: {Scab_Apple_resnet_batch}\nRot in batch: {Rot_Apple_resnet_batch}\nNormal in batch: {Normal_Apple_resnet_batch}\nBlotch in samples: {Blotch_Apple_resnet_samples}\nScab in samples: {Scab_Apple_resnet_samples}\nRot in samples: {Rot_Apple_resnet_samples}\nNormal in samples: {Normal_Apple_resnet_samples}\nGood: {good}\nBad: {bad}\nBad in batch: {number_of_bad_apples_resnet_batch}\nBad in samples: {number_of_bad_apples_resnet_samples}\nHossein bot:"

        # Get user question and send it to chatGPT function
        user_input = chat_bot_box.get()
        if user_input.lower() == "exit":
            bot_response = "Goodbye!"
        else:
            chat_prompt  = f'{initial_prompt} {user_input}\n'
            bot_response = chat_with_gpt3(chat_prompt)
        
        # Send bot response to interface
        gpt_bot_answer.set(f"GPT bot :  {bot_response}.")
    except ValueError:
        pass
    
# Function to create Team model chatbot
def team_chatbot():
    try:
        # Get input from user
        batch_size = int(batch_size_2.get())
        sample_size = int(sample_size_2.get())
        number_of_runs = int(nr_runs.get())
        aql = float(user_aql.get())

        # Set variables for chat bot
        name = "Make IT Work Team bot"
        good = 'Normal'
        bad = 'Rot + Blotch + Scab'
        sauce_factory_aql = 6.5
        syrup_factory_aql = 15

        # Generate random indices for batch size and sample size
        random.seed(f'{time.time() / 200:.1f}')
        random_numbers_batch = random.sample(range(len(y_pred_team)), batch_size)
        random_numbers_sample = random.sample(range(len(random_numbers_batch)), (sample_size * number_of_runs))

        # Create batch size with batch index and y_pred
        batch_size_team = [y_pred_team[index] for index in random_numbers_batch]

        # Create sample size with sample index and batch size
        samples_team = [batch_size_team[index] for index in random_numbers_sample]

        # Calculate the percentage of bad apples with team model
        if num_classes == 2:
            # For 2 classes
            Blotch_Apple_team_batch = 0
            Normal_Apple_team_batch = batch_size_team.count(0)
            Rot_Apple_team_batch = batch_size_team.count(1)
            Scab_Apple_team_batch = 0
            Blotch_Apple_team_samples = 0
            Normal_Apple_team_samples = samples_team.count(0)
            Rot_Apple_team_samples = samples_team.count(1)
            Scab_Apple_team_samples = 0

        else:
            # For 4 classes
            Blotch_Apple_team_batch = batch_size_team.count(0)
            Normal_Apple_team_batch = batch_size_team.count(1)
            Rot_Apple_team_batch = batch_size_team.count(2)
            Scab_Apple_team_batch = batch_size_team.count(3)
            Blotch_Apple_team_samples = samples_team.count(0)
            Normal_Apple_team_samples = samples_team.count(1)
            Rot_Apple_team_samples = samples_team.count(2)
            Scab_Apple_team_samples = samples_team.count(3)

        number_of_bad_apples_team_batch = (
                    Rot_Apple_team_batch + Blotch_Apple_team_batch + Scab_Apple_team_batch)
        number_of_bad_apples_team_samples = (
                    Rot_Apple_team_samples + Blotch_Apple_team_samples + Scab_Apple_team_samples)

        # Load the pre-trained sentence transformer model
        model = SentenceTransformer('all-MiniLM-L12-v2')

        # Define a dictionary of questions and answers
        qa_pairs = {
            "what is your name?": f" My name is {name}.",
            "can you tell me the total number of apples in the batch?": f"The total number of apples in the batch is {len(batch_size_team)}.",
            "can you tell me the total number of apples in the samples?": f"The total number of apples in the samples is {len(samples_team)}.",
            "can you tell me the total number of good apples in the batch?": f"The total number of apples in the batch is {Normal_Apple_team_batch}.",
            "can you tell me the total number of good apples in the samples?": f"The total number of apples in the samples is {Normal_Apple_team_samples}.",
            "can you tell me the total number of bad apples in the batch?": f"The total number of apples in the batch is {number_of_bad_apples_team_batch}.",
            "can you tell me the total number of bad apples in the samples?": f"The total number of apples in the samples is {number_of_bad_apples_team_samples}.",
            "how many good apples are in the batch?": f"The number of good apples in the batch is {Normal_Apple_team_batch}.",
            "how many good apples are in the samples?": f"The number of good apples in the samples is {Normal_Apple_team_samples}.",
            "how many bad apples are in the batch?": f"The number of bad apples in the batch is {number_of_bad_apples_team_batch}.",
            "how many bad apples are in the samples?": f"The number of bad apples in the samples is {number_of_bad_apples_team_samples}.",
            "what is the percentage of bad apples in the batch?": f"The percentage of bad apples in the batch is {number_of_bad_apples_team_batch / len(batch_size_team) * 100:.2f}%.",
            "what is the percentage of bad apples in the samples?": f"The percentage of bad apples in the samples is {number_of_bad_apples_team_samples / len(samples_team) * 100:.2f}%.",
            "how many bad apples are there in an approved batch?": f"The number of bad apples in an approved batch size of {len(batch_size_team)} is {number_of_bad_apples_team_batch}.",
            "how many bad apples are there in an approved samples?": f"The number of bad apples in an approved samples size of {len(samples_team)} is {number_of_bad_apples_team_samples}.",
            "how many apples are categorized as blotch in the batch?": f"The number of apples categorized as blotch is {Blotch_Apple_team_batch}.",
            "how many apples are categorized as blotch in the samples?": f"The number of apples categorized as samples is {Blotch_Apple_team_samples}.",
            "what is the proportion of rotten apples in the batch?": f"The proportion of rotten apples in the batch is {Rot_Apple_team_batch / len(batch_size_team):.2f}.",
            "what is the proportion of rotten apples in the samples?": f"The proportion of rotten apples in the samples is {Rot_Apple_team_samples / len(samples_team):.2f}.",
            "can we use this apple batch to Apple sauce factory?": f"Upon evaluation {'Yes, we can send this apple batch to Apple sauce factory.' if aql <= number_of_bad_apples_team_samples / len(samples_team) * 100 < 6.5 else 'No, there are not enough healthy apples.'}",
            "are there enough healthy apples to Apple syrup factory?": f"Upon evaluation {'Yes, there are enough healthy apples.' if 6.5 <= number_of_bad_apples_team_samples / len(samples_team) * 100 < 15 else 'No, there are not enough healthy apples.'}",
            "can we use this batch for the supermarket if the acceptance quality is increased by 1 percentage for the klasse 1?": f" {'Yes, it can be send to supermarket.' if number_of_bad_apples_team_samples / len(samples_team) * 100 <= aql + 1 else 'Even if the acceptance quality is increased by 1, it cannot be used in supermarket'}",
            "does the quality of the batch increase when the batch size is increased?": "The quality of the batch may or may not increase when the batch size is increased. It depends on various factors.",
            "what’s the average ratio between the healthy and unhealthy apples for different sample sizes?": f"The average ratio between healthy and unhealthy apples is {Normal_Apple_team_samples / number_of_bad_apples_team_samples}.",
        }

        # Calculate sentence embeddings for the questions
        question_embeddings = model.encode(list(qa_pairs.keys()))

        def get_fallback_answer():
            return "I'm sorry, but I don't have an answer to that question at the moment."
        
        def get_answer_2(user_query):
            query_embedding = model.encode([user_query]).flatten()
            similarities = [1 - cosine(query_embedding, q_emb) for q_emb in question_embeddings]
            max_similarity = max(similarities)
            most_similar_idx = similarities.index(max_similarity)

            if max_similarity >= 0.8:
                return qa_pairs[list(qa_pairs.keys())[most_similar_idx]]
            else:
                return get_fallback_answer()
            
        # Chat bot loop
        user_query = chat_bot_box.get().lower()
        if user_query == "exit":
            bot_response = "Goodbye!"
        else:
            chat_prompt = get_answer_2(user_query)            
            if chat_prompt is None:
                bot_response = get_fallback_answer()
            else:   
                bot_response = f"{chat_prompt}."        
            team_bot_answer.set(f"Team bot:  {bot_response}.")
            
    except ValueError:
        pass
    
# Creating method for adding resnet chart
def on_click():
    global my_img
    top = Toplevel()
    top.title('Pie chart')
    my_img = ImageTk.PhotoImage(Image.open('./../../data/charts/Pie_chart_ResNet.png'))
    Label(top, image=my_img).pack()

# Creating method for adding team model chart
def on_click_1():
    global my_img
    top = Toplevel()
    top.title('Pie chart')
    my_img = ImageTk.PhotoImage(Image.open('./../../data/charts/Pie_chart_Team.png'))
    Label(top, image=my_img).pack()
    
# Creating method for adding AQL chart information
def on_click_2():
    global my_img
    top = Toplevel()
    top.title('AQL chart')
    my_img = ImageTk.PhotoImage(Image.open('./../../data/charts/AQL_info_chart.jpeg'))
    Label(top, image=my_img).pack()
    
# Building interface
root = Tk()
root.title('Apple qualifier')

# Font properties
s = ttk.Style()
font_1 = ('Ariel Nova', 10)
s.configure('.', font = font_1)

mainframe = ttk.Frame(root, padding='3 3 12 12')
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)    

batch_size_2 = StringVar()
batch_size_entry = ttk.Entry(mainframe, width=10, font = font_1, textvariable=batch_size_2)
batch_size_entry.grid(column=2, row=1, sticky=(W, E))

sample_size_2 = StringVar()
sample_size_entry = ttk.Entry(mainframe, width=10, font = font_1, textvariable=sample_size_2)
sample_size_entry.grid(column=2, row=2, sticky=(W, E))

nr_runs = StringVar()
nr_runs_entry = ttk.Entry(mainframe, width=10, font = font_1, textvariable=nr_runs)
nr_runs_entry.grid(column=2, row=3, sticky=(W, E))

user_aql = StringVar()
user_aql_entry = ttk.Entry(mainframe, width=10, font = font_1, textvariable=user_aql)
user_aql_entry.grid(column=2, row=4, sticky=(W, E))

open_ai_access_key = StringVar()
user_aql_entry = ttk.Entry(mainframe, width=10, font = font_1, textvariable=open_ai_access_key)
user_aql_entry.grid(column=2, row=5, sticky=(W, E))

# Calculating Button (lower right) ResNet model
good_apple_percentage_resnet = StringVar()
ttk.Label(mainframe, textvariable=good_apple_percentage_resnet).grid(column=2, row=7, sticky=(W, E))

test_accuracy_resnet = StringVar()
ttk.Label(mainframe, textvariable=test_accuracy_resnet).grid(column=2, row=8, sticky=(W, E))

test_loss_resnet = StringVar()
ttk.Label(mainframe, textvariable=test_loss_resnet).grid(column=2, row=9, sticky=(W, E))

group_apple_category_resnet = StringVar()
ttk.Label(mainframe, width=30, textvariable=group_apple_category_resnet).grid(column=2, row=10, sticky=(W, E))

# Calculating Button (lower right) Team model
good_apple_percentage_team = StringVar()
ttk.Label(mainframe, textvariable=good_apple_percentage_team).grid(column=3, row=7, sticky=(W, E))

test_accuracy_team = StringVar()
ttk.Label(mainframe, textvariable=test_accuracy_team).grid(column=3, row=8, sticky=(W, E))

test_loss_team = StringVar()
ttk.Label(mainframe, textvariable=test_loss_team).grid(column=3, row=9, sticky=(W, E))

group_apple_category_team = StringVar()
ttk.Label(mainframe, width=30, textvariable=group_apple_category_team).grid(column=3, row=10, sticky=(W, E))

# Chatbot box
chat_bot_box = StringVar()
chat_bot_box_entry = ttk.Entry(mainframe, width=60, font = font_1, textvariable=chat_bot_box)
chat_bot_box_entry.grid(column=1, row=12, sticky=(W, E))

gpt_bot_answer = StringVar()
ttk.Label(mainframe, textvariable=gpt_bot_answer).grid(column=1, row=13, sticky=(W, E))

team_bot_answer = StringVar()
ttk.Label(mainframe, textvariable=team_bot_answer).grid(column=1, row=14, sticky=(W, E))

# Building interface Buttons
ttk.Button(mainframe, text='AQL information', command=on_click_2).grid(column=3, row=1, sticky=W)
ttk.Button(mainframe, text='      Run      ', command=calculate).grid(column=3, row=5, sticky=W)
ttk.Button(mainframe, text='ResNet18 Chart ', command=on_click).grid(column=2, row=11, sticky=W)
ttk.Button(mainframe, text='Team Chart     ', command=on_click_1).grid(column=3, row=11, sticky=W)
ttk.Button(mainframe, text='ChatGPT bot    ', command=gpt_chatbot).grid(column=2, row=12, sticky=W)
ttk.Button(mainframe, text='Team chatbot   ', command=team_chatbot).grid(column=3, row=12, sticky=W)

# Information Labels for Input Data Buttons
ttk.Label(mainframe, text=f"Batch Size (10-{len(y_pred_resnet)}): ").grid(column=1, row=1, sticky=W)
ttk.Label(mainframe, text="Sample Size (5-100):          ").grid(column=1, row=2,  sticky=W)
ttk.Label(mainframe, text="Number of Runs (1-100):       ").grid(column=1, row=3,  sticky=W)
ttk.Label(mainframe, text="Your AQL (1-100):             ").grid(column=1, row=4,  sticky=W)
ttk.Label(mainframe, text="Your OpenAI access key:       ").grid(column=1, row=5,  sticky=W)
ttk.Label(mainframe, text="    Without access key only Team chatbot works!").grid(column=1, row=6,  sticky=W)
ttk.Label(mainframe, text="The percentage of good apples:").grid(column=1, row=7,  sticky=W)
ttk.Label(mainframe, text="The test accuracy:            ").grid(column=1, row=8,  sticky=W)
ttk.Label(mainframe, text="The test loss:                ").grid(column=1, row=9,  sticky=W)
ttk.Label(mainframe, text="The apple category:           ").grid(column=1, row=10, sticky=W)
ttk.Label(mainframe, text="Please ask me a question:     ").grid(column=1, row=11, sticky=W)
ttk.Label(mainframe, text="Resnet18  ").grid(column=2, row=6, sticky=W)
ttk.Label(mainframe, text="Team model").grid(column=3, row=6, sticky=W)

# Interface INPUT loop and OUTPUT
for child in mainframe.winfo_children(): 
    child.grid_configure(padx=18, pady=9)    
    
batch_size_entry.focus()
sample_size_entry.focus()
nr_runs_entry.focus()
user_aql_entry.focus()
chat_bot_box_entry.focus()

root.bind('<Return>', calculate)
root.bind('<Return>', gpt_chatbot)
root.bind('<Return>', team_chatbot)

root.mainloop()
