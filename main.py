import tkinter as tk
from tkinter import ttk, messagebox, simpledialog , Toplevel, Label, Entry, Button, PhotoImage
import time
import re
from KenKen_Solvers.KenKen_Solver_Backtracking import solve_kenken as backtracking_algorithm
from KenKen_Solvers.KenKen_Solver_Genetic import genetic_algorithm

# Global variables
grid_size = 6  # Default
cages = []


class Cage:
    def __init__(self, cells, operation, target_value):
        self.cells = cells
        self.operation = operation
        self.target_value = target_value


def display_solution(solution, fitness=None, generation=None, performance=None):
    """Display the solved grid along with optional fitness, generation, and performance details."""
    solution_window = tk.Toplevel(root)
    solution_window.title("KenKen Solution")

    # Main frame for the display
    frame = ttk.Frame(solution_window, padding=10)
    frame.grid()

    # Stylish spacing for the solution details
    if fitness is not None or generation is not None or performance is not None:
        details_text = []
        if fitness is not None:
            details_text.append(f"Fitness: {fitness}")
        if performance is not None:
            details_text.append(f"Performance: {performance:.2f}%")
        
        combined_details = "   |   ".join(details_text)

        # Display the details with some spacing
        details_label = ttk.Label(frame, text=combined_details, font=("Arial", 12, "bold"), background="#D0F0C0", anchor="center")
        details_label.grid(row=0, column=0, columnspan=len(solution[0]), pady=(0, 15))  # Add vertical spacing below details

    # Display the grid with proper margin
    for i in range(len(solution)):
        for j in range(len(solution[i])):
            ttk.Label(frame, text=str(solution[i][j]), width=4, anchor="center",
                      borderwidth=2, relief="solid", background="#EAF6FF", font=("Arial", 12)).grid(
                row=i + 1, column=j  # Add spacing between grid cells
            )

    # Add a close button for convenience
    close_button = ttk.Button(frame, text="Close", command=solution_window.destroy)
    close_button.grid(row=len(solution) + 2, column=0, columnspan=len(solution[0]), pady=10)


def add_cage():
    """Enhanced Custom Cage Input Dialog with a User-Friendly Grid."""
    dialog = Toplevel(root)
    dialog.title("Add Cage")
    dialog.geometry("450x600")  # Adjusted for better proportions
    dialog.resizable(False, False)

    # Draw 6x6 Grid with Canvas
    cell_size = 50
    grid_canvas = tk.Canvas(dialog, width=6 * cell_size, height=6 * cell_size, bg="#f9f9f9", relief="ridge", bd=2)
    grid_canvas.grid(row=0, column=0, columnspan=2, pady=10)

    # Grid Lines & Coordinate Text
    for i in range(6):
        for j in range(6):
            x1, y1 = j * cell_size, i * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            grid_canvas.create_rectangle(x1, y1, x2, y2, outline="gray")
            grid_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=f"{i},{j}", font=("Arial", 9), fill="#666")

    # Add form fields with styled labels and inputs
    input_frame = tk.Frame(dialog)
    input_frame.grid(row=1, column=0, columnspan=2, pady=20)

    def add_input_row(label_text, row):
        label = Label(input_frame, text=label_text, font=("Arial", 10), anchor="w")
        label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        entry = Entry(input_frame, font=("Arial", 10), width=30, relief="groove", bd=2)
        entry.grid(row=row, column=1, padx=10, pady=5)
        return entry

    operation_entry = add_input_row("Operation (+, -, *, /)", 0)
    target_entry = add_input_row("Target Value", 1)
    cells_entry = add_input_row("Cells (e.g., (0,0), (0,1))", 2)


    # Submission Logic
    def submit():
        operation = operation_entry.get().strip()
        if operation not in ['+', '-', '*', '/']:
            messagebox.showerror("Error", "Invalid operation! Use +, -, *, or /")
            return

        try:
            target = int(target_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Target value must be an integer!")
            return

        cells_input = cells_entry.get().strip()
        try:
            cells = [tuple(map(int, m)) for m in re.findall(r'\((\d+),(\d+)\)', cells_input)]
            if not cells:
                raise ValueError()
        except:
            messagebox.showerror("Error", "Invalid cells format! Use (row,col) format.")
            return

        cages.append(Cage(cells, operation, target))
        messagebox.showinfo("Success", f"Cage '{operation} {target}' added successfully!")
        dialog.destroy()

    # Add Submit Button
    submit_button = Button(dialog, text="Submit", command=submit, font=("Arial", 10, "bold") , width=35, bg="#00539C", fg="white", bd=3)
    submit_button.grid(row=2, column=0, columnspan=2, pady=20)

    dialog.transient(root)
    dialog.grab_set()
    root.wait_window(dialog)

def reset_puzzle():
    """Reset the puzzle and cages to start fresh."""
    global cages
    cages = []  # Clear the cages list
    size_entry.delete(0, tk.END)
    size_entry.insert(0, "6")  # Default grid size
    algorithm_choice.set("Backtracking")  # Default algorithm choice
    messagebox.showinfo("Reset", "Puzzle and cages have been reset!")


def start_solving():
    """Start solving process."""
    global grid_size, cages
    try:
        grid_size = int(size_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Grid size must be an integer!")
        return

    algo = algorithm_choice.get()
    fitness = None
    generation = None
    performance = None
    start_time = time.time()
    if algo == "Backtracking":
        solution = backtracking_algorithm(grid_size, cages)
    else:
        fitness, generation, performance, solution = genetic_algorithm(cages, grid_size, 1000, 50, 0.7, 0.2)
    end_time = time.time()

    messagebox.showinfo("Success", f"Solution Found in {end_time - start_time:.2f}s")
    display_solution(solution , fitness, generation, performance)


def create_gui():
    """GUI Setup."""
    global root, size_entry, algorithm_choice

    root = tk.Tk()
    root.title("KenKen Puzzle Solver")
    root.geometry("600x400")
    root.configure(bg="#F5F7FA")

    title = tk.Label(root, text="KenKen Puzzle Solver", font=("Helvetica", 18, "bold"), bg="#00539C", fg="white", pady=20)
    title.pack(fill="x")

    # Input Frame
    input_frame = ttk.Frame(root, padding=30)
    input_frame.pack()

    ttk.Label(input_frame, text="Puzzle Size :", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=5)
    size_entry = ttk.Entry(input_frame, width=23 , justify='center')
    size_entry.grid(row=0, column=1, pady=10 , padx=5)
    size_entry.insert(0, "6")
    ttk.Separator()
    ttk.Label(input_frame, text="Algorithm :", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=5)
    algorithm_choice = ttk.Combobox(input_frame, values=["Backtracking", "Genetic Algorithm"], state="readonly" , justify='center', )
    algorithm_choice.grid(row=1, column=1, pady=5 , padx=5)
    algorithm_choice.set("Backtracking")

    ttk.Button(input_frame, text="Add Cage", command=add_cage , width=35).grid(row=2, column=0, columnspan=2,padx=10,pady=10)

    ttk.Button(input_frame, text="Solve Puzzle", command=start_solving, style="Accent.TButton" , width=35).grid(row=3, column=0, columnspan=2, pady=1)
    
    ttk.Button(input_frame, text="Reset Puzzle", command=reset_puzzle, width=35).grid(row=4, column=0, columnspan=2, pady=10)
    # Footer Section for Team ID
    footer = tk.Label(root, text="Team ID: 54", font=("Arial", 10, "bold"), bg="#00539C", fg="#FFFFFF", pady=10)
    footer.pack(side="bottom", fill="x")

    root.mainloop()


if __name__ == "__main__":
    create_gui()
