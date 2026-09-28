# employee_data_analysis

import numpy as np

experience = np.array([1, 3, 5, 2, 7, 4, 10, 6, 8, 2])
salary = np.array([25000, 35000, 50000, 28000, 65000, 45000, 90000, 60000, 75000, 30000])
rating = np.array([3.2, 4.1, 4.8, 3.5, 4.5, 3.9, 4.9, 4.2, 4.6, 3.7])

print(experience.shape,experience.dtype)
print(salary.shape,salary.dtype)
print(rating.shape,rating.dtype)

print(np.mean(experience))
print(np.mean(salary))
print(np.mean(rating))

print(salary[np.argmax(salary)])
print(np.argmax(salary))
print(salary[np.argmin(salary)])
print(salary[np.where(salary > 50000)])
print(rating[np.where(rating > 4.0)])
print(salary * 1.10)
print(np.max(salary) - np.min(salary))
print(np.median(salary))
print(np.std(salary))
print(np.sort(salary)[::-1])
print(np.where((experience > 5) &(salary > 60000),"True","False"))
print(salary[(experience > 5) & (salary > 60000)])

data = np.column_stack((experience, salary, rating))
print(data)
print(np.mean(data,axis=1))
print(data[np.argmax(data[:, 1])])

normalized_salary = (salary - np.min(salary)) / (np.max(salary) - np.min(salary))

print(normalized_salary)

"""
Conclusion: NumPy operations such as mean(), min(), max(),
argmax(), Boolean indexing, sorting, normalization,
and array manipulation were useful for analyzing the employee 
dataset.
"""