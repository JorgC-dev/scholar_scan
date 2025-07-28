from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.prompt import Prompt
from rich.table import Table
import time
import os
from train_model import KnnModelStudent
import tkinter as tk
from tkinter import messagebox


console = Console()

# ASCII code for intro
tittle = """
┌─────────────────────────────────────────────────────────────┐
│ ____       _           _              ____                  │
│/ ___|  ___| |__   ___ | | __ _ _ __  / ___|  ___ __ _ _ __  │
│\___ \ / __| '_ \ / _ \| |/ _` | '__| \___ \ / __/ _` | '_ \ │
│ ___) | (__| | | | (_) | | (_| | |     ___) | (_| (_| | | | |│  
│|____/ \___|_| |_|\___/|_|\__,_|_|    |____/ \___\__,_|_| |_|│
└─────────────────────────────────────────────────────────────┘
"""

# Actual dir
actual_dir = os.getcwd()
folder = actual_dir+'/25_07_25/'
fil_e = folder+'student_no_state.csv'
f_workloadChoices = ['Low','Middle','High']


def intro():
    """Intro of the programm"""
    console.clear()
    console.print(
        Panel(
            Align.center(
                f'[bold #27D3F5]{tittle}[/]', vertical="middle" 
            ),
            title="[bold blue] Ontario Tech University[/]",
            subtitle="[bold red] EdTech and Studen Analysis [/]",
            border_style="#F5276C dim", 
            padding=(1,5),
            style="on black"
        )
    )

    # Step 1: GET DATA AND TRAINING THE MODEL
    # training model starting
    with console.status("[bold yellow] AI is starting [/]", spinner="arc") as status: 
        status.update("[bold yellow] Getting the data [/]")
        time.sleep(2.0)
        # Get the data
        df = KnnModelStudent().get_data(fil_e)
        status.update("[bold yellow] Traininig model [/]")
        # train the model
        model,encoder, features_original = KnnModelStudent().train_model(df)
        time.sleep(2.0)
    # we create a panel to indicate AI is ready
    console.print("[bold green] ✔ Done! [/]")

    exitt = False
    while exitt == False:
        # DATA ENTRY 
        # title = Text("SAPAS | Student Attendance Prediction and Analysis System", style="bold yellow")
        # console.rule("[green]  [/]")
        console.clear()
        console.print(
            Panel(
                Align.center(f'[bold #EE2F92] AI is ready [/]', vertical="middle"),
                border_style="#21FDA1 dim",
                style="on black",
                padding=(1,5)
            ),
        )
        console.rule("[bold white] Enter the student's information below [/]")
        name = promptUses("Student name")
        gpa = promptUses("GPA(Grade Point Average)")

        # Table for information about Family workload referency
        table = Table()
        table.add_column("#")
        table.add_column("Family Workload Referency")
        # Add the workload choices
        for i, value in enumerate(f_workloadChoices):
            table.add_row(str(i+1),value)
        console.print(table)
        
        f_workload = Prompt.ask("[yellow]• Select the family workload: [/]", choices=[str(i) for i in range(1, len(f_workloadChoices)+1) ])
        s_hours = promptUses("Study hours:")

        s_fworkload = f_workloadChoices[int(f_workload)-1]

        # Send the parameters to another function
        response  = KnnModelStudent().predict_risk(gpa,s_fworkload,s_hours,encoder,model,features_original)

        console.print("\n")
        console.print("[bold green] ¿Student is likely to drop? [/]")
        # print the result
        table2 = Table()
        table2.add_column("Resut", style="magenta")
        table2.add_row(response)
        console.print(table2)

        # Then we ask to the user if want to exit to the program
        console.rule("")
        console.print("[bold cyan]\nDo you want to finish?[/bold cyan]")
        keyboard1 = Prompt.ask("[bold cyan]Press [bold red]E[/bold red] and then press [bold yellow] ENTER [/] to exit.\nPress [bold yellow] ENTER [/]to continue...[/bold cyan]")
        if keyboard1 == "E" or keyboard1=="E":
            console.clear()
            exitt = True
        else:
            exitt = False
            console.clear()

    console.print("\n[bold magenta]Thank you for uses our system![/bold magenta]")
    console.print(f"Here's what we collected:")

def promptUses(info_req):
    value = ""
    while value == "":
        value =  Prompt.ask(f'[yellow]• {info_req}[/yellow][red]*[/]')
    return value

def promptUses_choices(info_req):
    value = ""
    while value == "":
        value =  Prompt.ask(f'[yellow]• {info_req}[/yellow][red]*[/]')
    return value

intro()
