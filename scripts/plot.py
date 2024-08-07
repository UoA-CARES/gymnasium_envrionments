import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_csv_files(directory_path):
    # Define file paths
    train_file_path = os.path.join(directory_path, 'train.csv')
    eval_file_path = os.path.join(directory_path, 'eval.csv')

    # Check if files exist
    if not os.path.exists(train_file_path):
        print(f"Train file not found at {train_file_path}")
        return
    if not os.path.exists(eval_file_path):
        print(f"Eval file not found at {eval_file_path}")
        return

    # Read CSV files
    train_df = pd.read_csv(train_file_path)
    eval_df = pd.read_csv(eval_file_path)

    # Create subplots for train and eval data
    fig = px.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Train Data', 'Eval Data'))

    # Plot train data
    for col in train_df.columns:
        fig.add_trace(go.Scatter(x=train_df.index, y=train_df[col], mode='lines', name=f'Train {col}'), row=1, col=1)

    # Plot eval data
    for col in eval_df.columns:
        fig.add_trace(go.Scatter(x=eval_df.index, y=eval_df[col], mode='lines', name=f'Eval {col}'), row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=800, title_text="Train and Eval Data Plots")
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_csv.py <directory_path>")
    else:
        directory_path = sys.argv[1]
        plot_csv_files(directory_path)
