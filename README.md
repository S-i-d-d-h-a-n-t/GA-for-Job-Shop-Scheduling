# Job Shop Scheduling Optimization

This project implements a Genetic Algorithm (GA) solution for the Job Shop Scheduling Problem (JSSP). It aims to optimize production schedules by minimizing the makespan (total completion time) while satisfying job operation sequences and machine constraints.

## Features

### Genetic Algorithm Implementation:
- **Selection Mechanisms:** Tournament and random selection.
- **Crossover Operations:** Generate new solutions by combining existing ones.
- **Mutation Operations:** Introduce variations to avoid local optima.
- **Sequence Repair Mechanisms:** Ensure valid schedules after genetic operations.

### Visualization Tools:
- **Gantt Charts:** Visual representation of the schedule.
- **Performance Graphs:** Track makespan evolution across iterations.
- **Moving Average Filters:** Highlight trends in performance.

### Data Processing Capabilities:
- **Custom File Format Parser:** Read and process job scheduling data.
- **Job and Operation Handling:** Manage operations and their sequences.
- **Machine State Management:** Track machine availability and assignments.

## Requirements

The project requires the following Python libraries:

- matplotlib
- numpy
- random
- itertools
- json
- math
- time
- sys
- copy

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <https://github.com/S-i-d-d-h-a-n-t/GA-for-Job-Shop-Scheduling>
   ```

2. **Install Dependencies:**
   ```bash
   pip install matplotlib numpy
   ```

## Usage

1. **Prepare Input Data:**
   - Create an input file in the required format (see Input Format section).

2. **Run the Main Script:**
   ```bash
   python main.py
   ```

## Input Format

The input file should follow this structure:

- **First Line:** `<number_of_jobs> <number_of_machines>`
- **Subsequent Lines:** Job definitions with operations and machine assignments.

### Example:
```
2 3
2 3 1 3 2 2 3 4
1 2 1 4 3 1
```

## Configuration Parameters

Tune the genetic algorithm using these parameters:
- `populationSize:` Size of the population (default: 500).
- `tournamentSize:` Size of tournament selection (default: 400).
- `iterationSize:` Number of iterations (default: 1000).

## Key Functions

- **`parse(path):`** Parses input file and returns job and machine data.
- **`process_operations():`** Processes job operations and calculates makespan.
- **`generate_gantt_chart():`** Creates visual Gantt chart of the schedule.
- **`ga_operation():`** Main genetic algorithm implementation.

### Helper Functions:
- Population initialization.
- Selection mechanisms.
- Crossover and mutation operations.
- Schedule repair.
- Data visualization.

## Output

The program generates:

- **Optimized Operation Sequence:** Start and stop times for each operation.
- **Minimum Makespan Achieved:** Shortest total completion time.

### Performance Graphs:
- New makespans per iteration.
- Moving average of makespans.
- Minimum makespan over time.

### Gantt Chart:
- Visual representation of the final schedule.

## Performance Metrics

- **Execution Time:** Time taken to complete optimization.
- **Final Makespan:** Optimal schedule length.
- **Convergence Graphs:** Track algorithm progress.
- **Schedule Visualization:** Understand task distribution across machines.

## Contribution

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License

This project is open source and available under the MIT License.

## Contact

For questions and feedback, please open an issue in the project repository.
