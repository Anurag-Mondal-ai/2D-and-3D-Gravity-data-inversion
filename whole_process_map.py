import subprocess

# Display the options to the user
print("Enter 1. for Synthetic Model i.e. Model 1 of Sedimentary Basin for fixed Density.")
print("Enter 2. for Synthetic Model i.e. Model 1 of Sedimentary Basin for varying Density.")
print("Enter 3. for Synthetic Model i.e. Model 2 of Sedimentary Basin for fixed Density.")
print("Enter 4. for Synthetic Model i.e. Model 2 of Sedimentary Basin for varying Density.")
print("Enter 5. for Error Energy plot.")
print("Enter 6. for Prismatic Model for Model 1.")
print("Enter 7. for Prismatic Model for Model 2.")
print("Enter 8. for Real Model-Depth profile for Godavari Basin.")
print("Enter 9. for Real Model-Depth profile for San Jacinto Graben.\n")

# Prompt the user for input
while True:
    try:
        x = int(input("          Please Enter a number as per the option given above: "))
        script_map = {
            1: 'final_synthetic_model1.py',
            2: 'final_synthetic_model1_rho.py',
            3: 'final_synthetic_model2.py',
            4: 'final_synthetic_model2_rho.py',
            5: 'error_energy_plot.py',
            6: 'final_prism_model1_forward.py',
            7: 'final_prism_model2_forward.py',
            8: 'final_real_Zhou2012_3a.py',
            9: 'final_real_Chai1988_2a.py'
        }

        if x in script_map:
            subprocess.run(['python', script_map[x]])
            break
        else:
            print("Please enter a valid input\n")
    except ValueError:
        print("Invalid input. Please enter a number.\n")
