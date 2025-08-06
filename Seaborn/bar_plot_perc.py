# Create a bar plot of interest in math, separated by gender
sns.catplot(x="Gender", y="Interested in Math", kind="bar", data=survey_data)


# Show plot
plt.show()