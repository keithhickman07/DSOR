# Question 1:
# For Loop

myList = [1, 2, 3, 4]
newList = [1, 2, 3, 4]
addList = []

inputArray = []

for i in myList:
    for n in newList:
            if n > i:
                addList.append(n + i)

# List Comprehension
V = [i + n for i in myList for n in newList if n > i]

print(addList)
print(V)

# Question 2:
q2List = [1, 1, 2, 4, 5, 6, 7]
q2newList = [q2List]
q2

if q2List has duplicates:
    print "Yes"
    elif print "No"

