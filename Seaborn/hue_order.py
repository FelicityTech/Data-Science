# Import Pandas, Matplotlib, and Seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
student_data = pd.read_csv("student-alcohol-consumption.csv")

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location", hue_order=["Rural", "Urban"])

# Show plot
plt.show()