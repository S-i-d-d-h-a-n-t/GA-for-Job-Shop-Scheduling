import random
import itertools
import time
import sys
import json
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import time
%matplotlib inline

def parse(path):
    file = open(path, 'r')

    firstLine = file.readline()
    firstLineValues = list(map(int, firstLine.split()[0:2]))

    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    jobs = []

    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split()))

        operations = []

        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j]
            j = j+1

            operation = []

            for ik in range(k):
                machine = currentLineValues[j]
                j = j+1
                processingTime = currentLineValues[j]
                j = j+1

                operation.append({'machine': machine, 'processingTime': processingTime})

            operations.append(operation)

        jobs.append(operations)

    file.close()
    return {'machinesNb': machinesNb, 'jobs': jobs}


def noOfOptions(operation):
    return len(operation)

def noOfOperations(job):
    return len(job)

def noOfJobs(jobs):
    return len(jobs)

def totalNoOfOperations(jobs):
    operations = 0
    for job in jobs:
        operations = operations + noOfOperations(job)
    return operations

def totalNoOfOptions(jobs):
    options=1
    for job in jobs:
        for operation in job:
            options = options * noOfOptions(operation)
    return options

def rewriteJobs(jobs):
    for i,job in enumerate(jobs):
        for j,operation in enumerate(job):
            for k,option in enumerate(operation):
                # print(type(option))
                option['job'] = i
                option['operation'] = j
                option['option'] = k
    return jobs

def totalNoOfSequences(jobs):
    return math.factorial(totalNoOfOperations(jobs)) * totalNoOfOptions(jobs)

def getOperations(jobs):
    operations =[]
    for job in jobs:
        for operation in job:
            operations.append(operation)
    return operations

def getPrimitiveSequences(jobs):
    operations = getOperations(jobs)
    sequences = [[] for _ in range(len(operations))]
    for i, operation in enumerate(operations):
        for j, option in enumerate(operation):
            sequences[i].append(option)
    return sequences
    
def getSequencesShape(sequences):
    num_operations = len(s)
    max_options = max(len(operation) for operation in s)
    return (num_operations, max_options)  

def printSequences(sequences):
    for i, seq in enumerate(s):
        print(f"Sequence {i}: {seq}")

def check_sequence_constraint(operations):
    job_dict = {}
    for op in operations:
        job = op['job']
        if job not in job_dict:
            job_dict[job] = []
        job_dict[job].append(op)
    for job_ops in job_dict.values():
        for i in range(1, len(job_ops)):
            prev_op = job_ops[i-1]
            curr_op = job_ops[i]
            if prev_op['operation'] > curr_op['operation']:
                return False
    return True

def permute_operations(jobs):
    # Group the operations by job
    job_groups = {}
    for operation in jobs:
        job = operation['job']
        if job not in job_groups:
            job_groups[job] = []
        job_groups[job].append(operation)

    # Generate all permutations of operations for each job group
    permuted_groups = []
    for job, group in job_groups.items():
        permuted_groups.append(list(itertools.permutations(group)))

    # Combine the permuted job groups to generate all possible combinations
    combinations = list(itertools.product(*permuted_groups))

    # Flatten the combinations and return the result
    return [operation for job in combinations for operation in job]

def getValidSequencePermutations(sequencePermutations):
    result = []
    for perm in sequencePermutations:
        if(check_sequence_constraint(perm)):
            result.append(perm)
    return result

def getValidSequences(sequences):
    # Get all possible combinations of the sequences
    sequence_combinations = list((list(item) for item in itertools.product(*sequences) ))
    # print("\nAll possible Combinations:\n", sequence_combinations)
    # print("\nTotal possible Combinations:\n", len(sequence_combinations))

    result = []
    
    
    for combination in sequence_combinations:
        # print("\nCombination :", combination)
        sequence_permutations = list(list(item) for item in itertools.permutations(combination, r=None) )
        
        valid_sequence_permutations = getValidSequencePermutations(sequence_permutations)
        # print("\nPermutation Count:\n", len(valid_sequence_permutations))
        result.extend(valid_sequence_permutations)    
    
    return result

def process_operations(sequence, noOfJobs, noOfMachines):
    
    # initialize machine and job state
    machineState = {i: 0 for i in range(1, noOfMachines+1)}
    jobState = {i: 0 for i in range(noOfJobs)}
    #initialize job operation state 
    jobOpState = {i: 0 for i in range(noOfJobs)}
    #initialize operation state 
    opState = {i: False for i in range(len(sequence))}

    # print("Machine State:", machineState)
    # print("Job State:", jobState)
    # print("Job Operation State:", jobOpState)
    # print("Operation State:", opState)

    #initialize all output arrays
    opSequence = []
    startTimes = []
    stopTimes = []

    time=0
    loop = True
    possibleTimes = []
    makespan = 0
    while not all(opState.values()):
        executed = False  # flag to keep track if any operations were executed

        for i,op in enumerate(sequence):
            # print("Checking operation: ", i, " with opState: ", opState[i])
            if not(opState[i]) and op['operation'] == jobOpState[op['job']] and jobState[op['job']] <= time and machineState[op['machine']] <= time:
                # print("Executing Operation No.:", i)
                # set this operation is done
                opState[i] = True
                # output
                startTime = time
                stopTime = time + op['processingTime']
                opSequence.append(i)
                startTimes.append(startTime)
                stopTimes.append(stopTime)
                # intermediaries
                jobOpState[op['job']] += 1
                jobState[op['job']] = stopTime
                machineState[op['machine']] = stopTime
                possibleTimes.append(stopTime)
                executed = True  # set executed to True
                if stopTime>= makespan:
                    makespan = stopTime

        if not executed:  # if no operations were executed, increment time
            time += 1
        else:
            time = min(possibleTimes)  # if operations were executed, update time

    operationSequence = [None] * len(opSequence)  # Create a list of None values with length equal to opSequence
    for i, op_index in enumerate(opSequence):
        operationSequence[i] = sequence[op_index]  # Put the operation in its correct position in the list

    return makespan, operationSequence, startTimes, stopTimes 

def getAllData(sequences, noOfJobs, noOfMachines):
    makespans = []
    operationSequences = []
    allStartTimes = []
    allStopTimes = []
    for seq in sequences:
        makespan, operationSequence, startTimes, stopTimes = process_operations(seq, noOfJobs, noOfMachines)
        makespans.append(makespan)
        operationSequences.append(operationSequence)
        allStartTimes.append(startTimes)
        allStopTimes.append(stopTimes)
    return makespans, operationSequences, allStartTimes, allStopTimes
    
def generate_gantt_chart(sequence, start_times, stop_times,mk=0,reschtime=0):
    # Create a dictionary to store the colors for each job
    job_colors = {}
    for task in sequence:
        job_num = task['job']
        if job_num not in job_colors:
            job_colors[job_num] = plt.cm.tab20(len(job_colors))

    # Create the Gantt chart
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (18, 8)
    fig, gnt = plt.subplots()
    gnt.grid(True)

    # Set the y-axis labels to the machines
    gnt.set_yticks(range(1, max([task['machine'] for task in sequence]) + 1))
    gnt.set_yticklabels(['Machine {}'.format(i) for i in range(1, max([task['machine'] for task in sequence]) + 1)],fontweight="bold")

    # Set the x-axis limits to the minimum start time and maximum stop time
    gnt.set_xlim(min(start_times), max(stop_times))

    # Set the x-axis label to the time
    gnt.set_xlabel('Time\nMake-Span-%f'%mk,fontweight="bold")
    # plt.ylim([0, 20])
    # rescheduling time indicator
    gnt.axvline(x=reschtime, color='r',linewidth=3.0 ,linestyle='dashed')


    # Add each task to the Gantt chart as a rectangle
    for i,task in enumerate(sequence):
        job_num = task['job']
        color = job_colors[job_num]

        # Calculate the start and end times of the task
        start_time = start_times[i]
        end_time = stop_times[i]

        # Add the task to the Gantt chart
        gnt.broken_barh([(start_time, end_time - start_time)], (task['machine'], 1), facecolors=color, edgecolor='black')

        # Add the task label to the Gantt chart
        label = '({},{},{})'.format(task['job']+1, task['operation']+1, task['option']+1)
        gnt.text(start_time + (end_time - start_time) / 2, task['machine'] + 0.5, label, ha='center', va='center')

    # Show the Gantt chart

    plt.savefig('ganttsam.jpg')
    plt.show()

def printJobOperationInFormat(jobOperations):
    print("[", end='')
    for job in jobOperations:
        print("[", end='')
        for operation in job:
            print("[", end='')
            for option in operation:
                print("(", option['job'], ",",option['operation'], ",", option['option'], ")", end='')
            print("]", end='')
        print("]", end='')
    print("]")

def generateRandomSequence(jobOperations):
    sequence = []
    
    noOfOperations = totalNoOfOperations(jobOperations)
    # print("No of Jobs:", noOfJobs)
    # print("No of Operations:", noOfOperations) 
    # printJobOperationInFormat(jobOperations)


    for i in range(0,noOfOperations):
        randomJobNo = random.randrange(len(jobOperations))
        randomJob = jobOperations[randomJobNo]
        
        operation = randomJob.pop(0)
        # print("selected operation :", operation)
        randomOptionNo = random.randrange(len(operation))
        randomOption =  operation[randomOptionNo]
        sequence.append(randomOption)

        # print("---------------------------",i)
        # print("Job Selected:",randomJobNo)
        # print("Option Selected:",randomOptionNo)
        

        if len(operation) ==0:
            del randomJob[0]
        if len(randomJob) ==0:
            del jobOperations[randomJobNo]
    return sequence

def getInitialPopulation(populationSize, jobOperations):
    population = []

    for i in range(0, populationSize):
        randomSequence = generateRandomSequence(copy.deepcopy(jobOperations))
        population.append(randomSequence)
#         print("Chromosome size: ", len(randomSequence))

    return population

def random_selection(population,tournament_size):
    # Create a list of indices for the population
    indices = list(range(len(population)))
    
    # Shuffle the indices to randomly select a subset for each tournament
    random.shuffle(indices)
    
    # Initialize the selected parent indices to None
    parent1_index = None
    parent2_index = None

    # Iterate over the shuffled indices in tournament-size chunks
    for i in range(0, len(indices), tournament_size):
        # Get the indices of the individuals in the current tournament
        tournament_indices = indices[i:i+tournament_size]

        parents = random.sample(tournament_indices,2)
        parent1_index = parents[0]
        parent2_index = parents[1]
    
    # Return the selected parent indices as a tuple
    return (parent1_index, parent2_index)

def tournament_selection(population, makespans, tournament_size):
    """
    Select two parents from the population using tournament selection
    with the specified tournament size.

    Parameters:
    population (list): A list of individuals in the population, where each
                       individual is represented as a dictionary.
    makespans (list): A list of makespan values for each individual in the population.
    tournament_size (int): The size of each tournament.

    Returns:
    A tuple containing the indices of the selected parents in the population.
    """
    # Create a list of indices for the population
    indices = list(range(len(population)))
    
    # Shuffle the indices to randomly select a subset for each tournament
    random.shuffle(indices)
    
    # Initialize the selected parent indices to None
    parent1_index = None
    parent2_index = None
    
    # Iterate over the shuffled indices in tournament-size chunks
    for i in range(0, len(indices), tournament_size):
        # Get the indices of the individuals in the current tournament
        tournament_indices = indices[i:i+tournament_size]
        
        # Initialize the best fitness value and index to None
        best_fitness = None
        best_index = None
        
        # Iterate over the indices in the current tournament and find the individual with the best fitness
        for index in tournament_indices:
            fitness = makespans[index]
            if best_fitness is None or fitness < best_fitness:
                best_fitness = fitness
                best_index = index
                
        # Update the selected parent indices if necessary
        if parent1_index is None:
            parent1_index = best_index
        elif parent2_index is None:
            parent2_index = best_index
        else:
            # Both parents have been selected, so break out of the loop
            break
    
    # Return the selected parent indices as a tuple
    return (parent1_index, parent2_index)

def delete_operation(chromosome, job_no, operation_no):
    for i in range(len(chromosome)):
        if chromosome[i]['job'] == job_no and chromosome[i]['operation'] == operation_no:
            del chromosome[i]
            break

def crossover(parent1, parent2):
    # Select random operations from parent 1
    num_operations = len(parent1)
    # print("parent1: ", num_operations)
    num_selected = random.randint(1, num_operations)
    selected_indices = random.sample(range(num_operations), num_selected)

    # Create child by adding selected operations from parent 1
    child = []
    for i in range(num_operations):
        if i in selected_indices:
            operation = parent1[i]
            child.append(operation)
        

    # Delete selected operations from parent 2
    for operation in child:
        if operation is not None:
            job, operation_no = operation['job'], operation['operation']
            delete_operation(parent2, job, operation_no)

    # Add remaining operations from parent 2 to child
    child.extend(parent2)

    return child

def mutation(chromosome):
    """
    Perform mutation on a chromosome by swapping the positions of two operations.

    Args:
        chromosome (list): A list of operations representing the chromosome.

    Returns:
        list: The mutated chromosome.
    """
    # Select two random operations to swap
    pos1 = random.randint(0, len(chromosome) - 1)
    pos2 = random.randint(0, len(chromosome) - 1)
    while pos1 == pos2:  # Ensure that the two positions are different
        pos2 = random.randint(0, len(chromosome) - 1)

    # Swap the positions of the two operations
    chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]

    return chromosome

def repair(child):
    repairedChild = []
    jobPositions = []

    for operation in child:
        jobPositions.append(operation['job'])

    # Sort operations for each job based on operation number
    jobs = {}
    for operation in child:
        job = operation['job']
        if job not in jobs:
            jobs[job] = []
        jobs[job].append(operation)
    for job in jobs.values():
        job.sort(key=lambda x: x['operation'])

    # Check and correct sequence constraint
    for i,jobNo in enumerate(jobPositions):
        repairedChild.append(jobs[jobNo].pop(0))

    return repairedChild

def replace_worst_with_child(population, makespans, child, child_makespan):
    worst_index = makespans.index(max(makespans))
    if child_makespan <= makespans[worst_index]:
        population[worst_index] = child
        makespans[worst_index] = child_makespan

    
def plot_new_makespans(values, title):
    """
    Plots a line graph of the input values.

    Parameters:
    values (list): A list of numerical values.

    Returns:
    None
    """
    # Generate x-axis values
    x_values = list(range(len(values)))

    # Create a new plot
    plt.plot(x_values, values)

    # Set the x-axis label
    plt.xlabel('Iteration')

    # Set the y-axis label
    plt.ylabel('Value')

    # Add a title to the plot
    plt.title(title)

    # Set the y-axis limits
    # plt.ylim([0, max(values)])

    # Display the plot
    plt.show()
    
def plot_new_makespans(values, title):
    """
    Plots a line graph of the input values.

    Parameters:
    values (list): A list of numerical values.

    Returns:
    None
    """
    # Generate x-axis values
    x_values = list(range(len(values)))

    # Create a new plot
    plt.plot(x_values, values)

    # Set the x-axis label
    plt.xlabel('Iteration')

    # Set the y-axis label
    plt.ylabel('Value')

    # Add a title to the plot
    plt.title(title)

    # Set the y-axis limits
    # plt.ylim([0, max(values)])

    # Display the plot
    plt.show()

def kalman_filter(data, level_of_smoothening):
    """
    Applies Kalman filter to smooth the given data stream.

    Parameters:
    data (list): A list of data points to be filtered.
    level_of_smoothening (int): An integer indicating the level of smoothening.

    Returns:
    A numpy array of smoothed data points.
    """
    # Define the initial state matrix and covariance matrix
    initial_state = np.array([data[0], 0])
    initial_covariance = np.identity(2)

    # Define the process noise matrix and measurement noise matrix
    process_noise = np.array([[0.1 * level_of_smoothening, 0], [0, 0.1 * level_of_smoothening]])
    measurement_noise = np.array([0.1 * level_of_smoothening])

    # Define the state transition matrix and measurement matrix
    state_transition = np.array([[1, 1], [0, 1]])
    measurement = np.array([1, 0])

    # Initialize the state and covariance matrices
    state = initial_state
    covariance = initial_covariance

    # Initialize an array to hold the smoothed data points
    smoothed_data = []

    # Apply the Kalman filter to each data point
    for point in data:
        # Predict the next state using the state transition matrix and process noise matrix
        predicted_state = state_transition @ state
        predicted_covariance = state_transition @ covariance @ state_transition.T + process_noise

        # Compute the Kalman gain using the predicted covariance matrix and measurement matrix
        kalman_gain = predicted_covariance @ measurement.T / (measurement @ predicted_covariance @ measurement.T + measurement_noise)

        # Update the state and covariance matrices using the Kalman gain and the difference between the measured value and the predicted value
        state = predicted_state + kalman_gain * (point - measurement @ predicted_state)
        covariance = (np.identity(2) - kalman_gain @ measurement) @ predicted_covariance

        # Add the smoothed data point to the array
        smoothed_data.append(state[0])

    # Convert the smoothed data array to a numpy array
    smoothed_data = np.array(smoothed_data)

    return smoothed_data

def moving_average_filter(data, window_size):
    """
    Applies moving average filter to smooth the given data stream.

    Parameters:
    data (list): A list of data points to be filtered.
    window_size (int): Size of the window to be used for smoothing.

    Returns:
    A numpy array of smoothed data points.
    """
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='valid')
    return smoothed_data

def generation_makespan(value):
    plt.plot([i for i in range(len(value))],value,'b')
    plt.ylabel('makespan',fontsize=15)
    plt.xlabel('generation',fontsize=15)
    plt.savefig('itersam1.jpg')
    plt.show()

def addrewriteJobs(jobs,noofjobs):
    newnoofjobs = noofjobs-1
    for i,job in enumerate(jobs):
        newnoofjobs = newnoofjobs+1
        for j,operation in enumerate(job):
            for k,option in enumerate(operation):
                # print(type(option))
                option['job'] = newnoofjobs
                option['operation'] = j
                option['option'] = k
    return jobs


def completion_time(sequence, noOfJobs, noOfMachines):
    
    # initialize machine and job state
    machineState = {i: 0 for i in range(1, noOfMachines+1)}
    jobState = {i: 0 for i in range(noOfJobs)}
    #initialize job operation state 
    jobOpState = {i: 0 for i in range(noOfJobs)}
    #initialize operation state 
    opState = {i: False for i in range(len(sequence))}

    # print("Machine State:", machineState)
    # print("Job State:", jobState)
    # print("Job Operation State:", jobOpState)
    # print("Operation State:", opState)

    #initialize all output arrays
    opSequence = []
    startTimes = []
    stopTimes = []

    time=0
    loop = True
    possibleTimes = []
    makespan = 0
    while not all(opState.values()):
        executed = False  # flag to keep track if any operations were executed

        for i,op in enumerate(sequence):
            # print("Checking operation: ", i, " with opState: ", opState[i])
            if not(opState[i]) and op['operation'] == jobOpState[op['job']] and jobState[op['job']] <= time and machineState[op['machine']] <= time:
                # print("Executing Operation No.:", i)
                # set this operation is done
                opState[i] = True
                # output
                startTime = time
                stopTime = time + op['processingTime']
                opSequence.append(i)
                startTimes.append(startTime)
                stopTimes.append(stopTime)
                # intermediaries
                jobOpState[op['job']] += 1
                jobState[op['job']] = stopTime
                machineState[op['machine']] = stopTime
                possibleTimes.append(stopTime)
                executed = True  # set executed to True
                if stopTime>= makespan:
                    makespan = stopTime

        if not executed:  # if no operations were executed, increment time
            time += 1
        else:
            time = min(possibleTimes)  # if operations were executed, update time
#     print("machineState",machineState)
#     print("jobState",jobState)
#     print("jobOpState",jobOpState)
    operationSequence = [None] * len(opSequence)  # Create a list of None values with length equal to opSequence
    for i, op_index in enumerate(opSequence):
        operationSequence[i] = sequence[op_index]  # Put the operation in its correct position in the list

    return machineState, jobState, jobOpState, opState, opSequence, startTimes, stopTimes, possibleTimes, operationSequence

def ga_operation(jobs,noOfJobs,noOfMachines):
    
    t0 = time.time()

    populationSize = 500
    tournamentSize = 400
    iterationSize = 1000

    population = getInitialPopulation(populationSize, jobs)

    for i, p in enumerate(population):
        if not check_sequence_constraint(p):
            print("invalid")

    makespans, operationSequences, allStartTimes, allStopTimes = getAllData(population, noOfJobs, noOfMachines)

    # print("\n\n********************************   STEP 3 - Selection by Tournament, Random Crossover, Sequence Mutation, Job Sequence Repairment  *********************************\n")

    newMakeSpans = []
    minMakeSpans = []

    for i in range(0, iterationSize):
    #     print("Iteration: ", i)
        # Tournament Selection of 2 parents
    #     parent1, parent2 = tournament_selection(population, makespans, tournamentSize)

        # random Selection of 2 parents
        parent1, parent2 = random_selection(population,tournamentSize)
        # Crossover to generate Child
        crossoverChild  = crossover(copy.deepcopy(population[parent1]), copy.deepcopy(population[parent2]))
        # print("\nCrossover Child: ", crossoverChild)

        # Mutation in Child
        mutatedChild = mutation(crossoverChild)
        # print("\nMutation Child: ", mutatedChild)

        # Repair Child for job operation constraint
        repairChild = repair(mutatedChild)
        # print("\nRepair Child: ", repairChild)
        # time.sleep(0.1)

        # Get Makespan of Child
        makespan, operationSequence, startTimes, stopTimes = process_operations(repairChild, noOfJobs, noOfMachines)

        newMakeSpans.append(makespan)

        # Replace Child with worst chromosome in population
        replace_worst_with_child(population, makespans, repairChild, makespan)    ###changing orignal population
        minMakeSpans.append(min(makespans))

    # print("\n\n********************************   STEP 4 - Minimum Makespan Sequence of Population and Gantt Chart *********************************\n")

    ### finding index of minimum makespan in population 
    minimumMakespan = min(makespans)
    indexOfMinimumMakespan = makespans.index(minimumMakespan)
    
    makespan, operationSequence, startTimes, stopTimes = process_operations(population[indexOfMinimumMakespan], noOfJobs, noOfMachines)
    print("\nMinimum Makespan: ",minimumMakespan)
    print("\nIndex of Minimum Makespan: ",indexOfMinimumMakespan)

    t1 = time.time()
    total_time = t1 - t0
    print("\nFinished in {0:.2f}s".format(total_time))

    print("\nMakeSpan:", makespan)
    print("\nStartTimes:", startTimes)
    print("\nStopTimes:", stopTimes)
    print("\nOperation Sequence:", operationSequence)
#     print("\nAll New Makespans: ", newMakeSpans)
    print("\nCount New Makespans: ", len(newMakeSpans))
    
    return operationSequence,startTimes,stopTimes,minMakeSpans,newMakeSpans,makespan


print("\n\n****************1st batch dataset processing*****************************\n")

####  1st batch dataset
data = parse("test_data\sf2batch1.txt")
noOfMachines = data['machinesNb']
jobs = data['jobs']
noOfJobs = len(jobs)
print('Number of machines:', noOfMachines)
print('Number of jobs:', noOfJobs)
jobs = rewriteJobs(jobs)
jobs

operationSequence1,startTimes1,stopTimes1,minMakeSpans1,newMakeSpans1,makespan1 = ga_operation(jobs,noOfJobs,noOfMachines)

# Plot the graph
plot_new_makespans(newMakeSpans1, "All New MakeSpans")

filteredData = moving_average_filter(newMakeSpans1,100)
plot_new_makespans(filteredData, "Moving Average of 100")

minMakeSpansData = moving_average_filter(minMakeSpans1,100)
plot_new_makespans(minMakeSpansData, "Minimum MakeSpan over time")

generation_makespan(minMakeSpans1)

generate_gantt_chart(operationSequence1, startTimes1, stopTimes1,makespan1)