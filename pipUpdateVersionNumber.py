# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:01:45 2019

@author: Patrick
"""



""" 
Set name of the file to update. Should be setup.py unless you have a specfic
reason not to.
"""
filename = 'setup.py'


"""
Read file.
"""
with open(filename, 'r') as f:
    content = f.read()


"""
Extract version number with some beautiful list comprehension magic.
"""
versionLine = [line for line in content.split('\n') if "version" in line][0]
versionString = [word for word in versionLine.split('"')][1]
version = [int(number) for number in versionString.split('.')]


"""
Prompt user on the type of update. The options are:
    major - Increment first number of version by 1 and set the other two to 0.
    minor - Increment second number of version by 1 and set last one to 0.
    bugfix - Increment last number of version
Default behavior assumes bugfix as input.
"""
updateType = input('Is this a major update, a minor update or a bugfix? : ')


if updateType.lower() == 'major':
    version[0] = version[0] + 1
    version[1] = 0
    version[2] = 0
    
elif updateType.lower() == 'minor':
    version[1] = version[1] + 1
    version[2] = 0

else:
    version[2] = version[2] + 1
    
newVersionString = str(version[0]) + '.' + str(version[1]) + '.' + str(version[2])

print( 'Updating version number in ' + filename + ' to ' + newVersionString )


"""
Create new string from the original, updating version number.
"""
newContent = ''

for line in content.split('\n'):
    if "version" in line:
        for ii, word in enumerate(versionLine.split('"')):
            if ii == 1:
                newContent += newVersionString
            else:
                newContent += word
            newContent += '"'
        newContent = newContent[0:-1]
    else:
        newContent += line
    newContent += '\n'
newContent = newContent[0:-1]  


"""
Overwrite file's content with computed newContent.
"""
with open(filename, 'w+') as f:
    f.write(newContent)
    


    
    
    
    
    