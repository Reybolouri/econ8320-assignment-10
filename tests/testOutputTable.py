import unittest
import json
import sys as system
import io
import pandas as pd
import re
import numpy as np


with open("Lesson.ipynb", "r") as file:
    f_str = file.read()

doc = json.loads(f_str)

code = [i for i in doc['cells'] if i['cell_type']=='code']
si = {}
for i in code:
    for j in i['source']:
        if "#si-exercise" in j:
            exec(compile("".join(i['source']), '<string>', 'exec'))


class TestCase(unittest.TestCase):

    def testSummaryTableOutput(self):
        stdout = system.stdout
        system.stdout = io.StringIO()

        data = pd.read_csv("tests/files/assignment8Data.csv")
        x = data.loc[:100, ['sex', 'age', 'educ']]
        y = data.loc[:100, 'white']
        reg = RegressionModel(x, y, create_intercept=True, regression_type='logit')
        reg.fit_model()
        reg.summary()

        output = system.stdout.getvalue()
        system.stdout = stdout

        print("Captured Summary Table Output:")
        print(output)

        labels = bool(re.findall(r'[Cc]oef.*[Ss]t.*[Zz].*[Pp].*', output))
        sex = bool(re.findall(r'sex.*\d+.*\d+.*\d+.*\d+.*', output))
        age = bool(re.findall(r'age.*\d+.*\d+.*\d+.*\d+.*', output))
        educ = bool(re.findall(r'educ.*\d+.*\d+.*\d+.*\d+.*', output))
        intercept = bool(re.findall(r'intercept.*\d+.*\d+.*\d+.*\d+.*', output))

        print("Check for label presence:", labels)
        print("Check for sex row:", sex)
        print("Check for age row:", age)
        print("Check for educ row:", educ)
        print("Check for intercept row:", intercept)

        self.assertTrue(labels & sex & age & educ & intercept, "Your table is not correctly formatted.")
