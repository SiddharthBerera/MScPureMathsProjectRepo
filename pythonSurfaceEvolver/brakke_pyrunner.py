import subprocess

# Path to  Surface Evolver `.fe` file
evolver_file = "catbody.fe"  

# Start Surface Evolver as a subprocess with text-based interaction
evolver = subprocess.Popen(
    ["evolver", evolver_file],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True  
)

# Send a command to Evolver and read output
def send_command(command):
    """ Sends a command to Evolver and reads the response """
    evolver.stdin.write(command + "\n")  # Send command
    evolver.stdin.flush()  # Ensure the command is processed
    output = []
    while True:
        line = evolver.stdout.readline().strip()
        if not line:  # Stop reading when there's no more output
            break
        output.append(line)
    return "\n".join(output)

# Interactive loop to send commands dynamically
print("Interactive Surface Evolver CLI")
print("Type Evolver commands (e.g., 'r' to refine, 'g 100' to evolve 100 steps).")
print("Type 's' to view the interactive graphics. Type 'exit' to quit.\n")

while True:
    command = input("Enter command: ").strip()
    
    if command.lower() == "exit":
        send_command("q")  # Exit Evolver
        print("Exiting Evolver...")
        break
    elif command.lower() == "s":
        print("Opening interactive graphics...")
        send_command("s")  # Open the interactive graphics window
    else:
        output = send_command(command)
        print(output)

# Close the Evolver process
evolver.stdin.close()
evolver.stdout.close()
evolver.stderr.close()
evolver.wait()
