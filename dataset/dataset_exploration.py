import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def create_dataframe(ages, genders, races):
    """
    Create a pandas DataFrame from ages, genders, and races lists.
    Adds validation to ensure all lists have the same length.
    """
    if not (len(ages) == len(genders) == len(races)):
        raise ValueError("The input lists must have the same length.")
    
    data = {'age': ages, 'gender': genders, 'race': races}
    return pd.DataFrame(data)

def plot_age_distribution_by_gender(df):
    """
    Plot the age distribution for male and female groups.
    Uses histograms with KDE overlays to visualize distributions.
    """
    if df.empty:
        print("DataFrame is empty. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Age Distribution by Gender', fontsize=20)

    # Separate data based on gender
    df_male = df[df['gender'] == 0]
    df_female = df[df['gender'] == 1]

    # Plot histograms and KDEs
    sns.histplot(df_male['age'], kde=True, color="blue", bins=50, ax=axes[0])
    sns.histplot(df_female['age'], kde=True, color="orange", bins=50, ax=axes[1])
    
    axes[0].set_title("Male")
    axes[1].set_title("Female")

    sns.kdeplot(df_male['age'], color="blue", ax=axes[2], label="Male", shade=True)
    sns.kdeplot(df_female['age'], color="orange", ax=axes[2], label="Female", shade=True)
    axes[2].legend()
    axes[2].set_title("Male vs Female")

    plt.tight_layout()
    plt.show()

def plot_gender_race_pie_charts(df):
    """
    Plot pie charts showing gender and race distribution.
    Ensures labels and colors are appropriately set and the plot is accessible.
    """
    if df.empty:
        print("DataFrame is empty. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Gender distribution pie chart
    gender_counts = df['gender'].value_counts()
    gender_labels = ["Male", "Female"]
    gender_colors = sns.color_palette("Set2")
    
    gender_counts.plot(kind='pie', labels=gender_labels, autopct='%1.1f%%', 
                       startangle=90, colors=gender_colors, ax=axes[0])
    axes[0].set_ylabel('')
    axes[0].set_title('Gender Distribution')

    # Race distribution pie chart
    race_counts = df['race'].value_counts()
    race_labels = ["White", "Black", "Asian", "Indian", "Other"]
    race_colors = sns.color_palette("Set3")

    race_counts.plot(kind='pie', labels=race_labels, autopct='%1.1f%%', 
                     startangle=90, colors=race_colors, ax=axes[1])
    axes[1].set_ylabel('')
    axes[1].set_title('Race Distribution')

    plt.tight_layout()
    plt.show()

def plot_age_distribution_by_race(df):
    """
    Plot age distributions for different races using histograms and KDE overlays.
    Ensures each plot is color-coded and properly labeled.
    """
    if df.empty:
        print("DataFrame is empty. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle('Age Distribution by Race', fontsize=20)

    race_groups = {
        0: ("White", "red"),
        1: ("Black", "orange"),
        2: ("Asian", "blue"),
        3: ("Indian", "green"),
        4: ("Other", "purple")
    }

    # Loop through race groups and create plots
    for i, (race_id, (race_name, color)) in enumerate(race_groups.items()):
        df_race = df[df['race'] == race_id]
        sns.histplot(df_race['age'], kde=True, color=color, bins=40, ax=axes[i])
        axes[i].set_title(f"{race_name} (Mean Age: {df_race['age'].mean():.2f})")

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    plt.tight_layout()
    plt.show()

# Example usage:
ages = [23, 35, 45, 20, 60, 34, 28, 18]  # Replace with actual data
genders = [0, 1, 0, 1, 0, 1, 0, 1]  # Replace with actual gender labels (0 for male, 1 for female)
races = [0, 1, 2, 0, 3, 2, 4, 0]  # Replace with actual race labels

# Create the DataFrame
df = create_dataframe(ages, genders, races)

# Plotting the distributions
plot_age_distribution_by_gender(df)
plot_gender_race_pie_charts(df)
plot_age_distribution_by_race(df)
