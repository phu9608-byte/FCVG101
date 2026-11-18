"""FCVG101 Task 3 - FruitSort Vision Assistant.

Modules
-------
settings : paths to the Task 1 / Task 2 code and artifacts, constants
errors   : AssistantError (message + how-to-fix hint shown to the user)
engines  : loads the Task 1 engine (task1/src) and the Task 2 model (task2/src)
intent   : recognises the requested task (and parameters) from a free-text request
gradcam  : Grad-CAM heat map for the Task 2 CNN
tools    : one function per mode; each calls Task 1 / Task 2 code and returns a ToolResult
"""
