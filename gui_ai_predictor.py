import time
import os
from train_model import KnnModelStudent
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


# Actual dir
actual_dir = os.getcwd()
folder = actual_dir+'/25_07_25/'
fil_e = folder+'student_no_state.csv'
f_workloadChoices = ['Low','Middle','High']


def screen():
    # main window for the application
    window = tk.Tk()
    window.title("Scholar Scan")
    # window.geometry("400x500") # Width x High

    width_window = window.winfo_screenwidth()
    high_window = window.winfo_screenheight()
    x_pos = (width_window // 2) - (400 // 2)
    y_pos = (high_window // 2) - (600 // 2)

    # 3. Aplicar la geometría a la ventana
    window.geometry(f'{400}x{600}+{x_pos}+{y_pos}')
    # Widgets 

    # --- Etiqueta y Campo de Entrada para el Nombre ---
    lbl_name = tk.Label(window, text="Student name:", font=("Arial", 12))
    lbl_name.pack(pady=(20, 5)) # Padding superior e inferior

    global input_name
    input_name = tk.Entry(window, width=40, font=("Arial", 12))
    input_name.pack(pady=5)

    # --- Etiqueta y Campo de Entrada para el Mensaje ---
    lbl_gpa = tk.Label(window, text="Grade Point Average(GPA):", font=("Arial", 12))
    lbl_gpa.pack(pady=5)

    global input_gpa
    input_gpa = tk.Entry(window, width=40, font=("Arial", 12))
    input_gpa.pack(pady=5)

    # --- Label & Combobox for Family Workload ---
    lbl_fwad = tk.Label(window, text="Family Workload:", font=("Arial", 12))
    lbl_fwad.pack(pady=5)

    global combo_fwad
    combo_fwad = ttk.Combobox(window, values=f_workloadChoices, state="readonly", font=("Arial", 12), width=37)
    combo_fwad.pack(pady=5)
    combo_fwad.current(0)  # Set default selection to first option

    # --- label & input to study hours ---
    lbl_stdy = tk.Label(window, text="Study hours:", font=("Arial", 12))
    lbl_stdy.pack(pady=5)

    global input_stdy
    input_stdy = tk.Entry(window, width=40, font=("Arial", 12))
    input_stdy.pack(pady=5)

    global saludo_variable
    saludo_variable = tk.StringVar()
    saludo_variable.set("") 
    etiqueta_personalizada = tk.Label(
        window,
        textvariable=saludo_variable,
        fg="white",           # Color del texto: blanco
        bg="#0088BA",         # Color de fondo: azul Bootstrap
        font=("Helvetica", 16, "bold"), # Fuente, tamaño y estilo
        padx=20,              # Relleno horizontal
        pady=10,              # Relleno vertical
        relief="solid",       # Borde sólido
        bd=2                  # Ancho del borde de 2 píxeles
    )
    etiqueta_personalizada.pack(pady=50)

    # Valor inicial

    # --- Botones ---
    # Botón para mostrar los datos
    boton_mostrar = tk.Button(window, text="Predict", command=predict,
                            font=("Arial", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
    boton_mostrar.pack(pady=15) # Padding vertical

    # Botón para limpiar los campos
    boton_limpiar = tk.Button(window, text="Delete fields", command=limpiar_campos,
                            font=("Arial", 12), bg="#f44336", fg="white", activebackground="#da190b")
    boton_limpiar.pack(pady=5)

    # Fotter
    lbl_about = tk.Label(window, text="Scholar Scan© 2025 | Ontario Tech University ", font=("Arial", 12))
    lbl_about.pack(pady=5)

    # 3. Iniciar el bucle principal de la aplicación
    window.mainloop()

def predict():
    """
    Funcion que pone en funcionamiento la prediccion de modelos
    """
    name =  input_name.get()
    gpa = input_gpa.get()
    combo = combo_fwad.get()
    study = input_stdy.get()

    # Turn on the machine learnig to running
    # Get the data
    df = KnnModelStudent().get_data(fil_e)

    # train the model
    model,encoder, features_original = KnnModelStudent().train_model(df)

    # Send the parameters to another function
    response  = KnnModelStudent().predict_risk(gpa,combo,study,encoder,model,features_original)
    messagebox.showinfo("Response", f"Student: {name}\nthe student is likely to drop? {response}")
    saludo_variable.set(f'The  student is likely  to dro out?: {response}')

def limpiar_campos():
    """
    Función para limpiar los campos de entrada.
    """
    input_name.delete(0, tk.END)
    input_gpa.delete(0, tk.END)
    input_stdy.delete(0, tk.END)
    input_stdy.delete(0, tk.END)

screen()