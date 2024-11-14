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

    def testLogitCorrectValues(self):
        data = pd.read_csv("tests/files/assignment8Data.csv")
        x = data.loc[:100, ['sex', 'age', 'educ']]
        y = data.loc[:100, 'white']
        reg = RegressionModel(x, y, create_intercept=True, regression_type='logit')
        reg.fit_model()

        sex = {'coefficient': -1.1229156890097627,
               'standard_error': 0.39798772782618025,
               'z_stat': -2.821483202869492,
               'p_value': 0.004780214077269219}
        age = {'coefficient': -0.007012518056833769,
               'standard_error': 0.010835821823286998,
               'z_stat': -0.6471607019011091,
               'p_value': 0.5175279421902776}
        educ = {'coefficient': -0.046485475816343394,
                'standard_error': 0.10100278092776117,
                'z_stat': -0.46023956359766527,
                'p_value': 0.6453442758780246}
        intercept = {'coefficient': 5.735435005488546,
                     'standard_error': 1.1266207023561843,
                     'z_stat': 5.090830475148922,
                     'p_value': 3.56498650369634e-07}

        print("Testing Logit Model Coefficients and Statistics for Sex")
        print("Expected:", sex)
        print("Computed:", reg.results['sex'])

        sexEq = (round(sex['coefficient'] - reg.results['sex']['coefficient'], 1) == 0) & \
                (round(sex['standard_error'] - reg.results['sex']['standard_error'], 1) == 0) & \
                (round(sex['z_stat'] - reg.results['sex']['z_stat'], 1) == 0) & \
                (round(sex['p_value'] - reg.results['sex']['p_value'], 1) == 0)

        print("Testing Logit Model Coefficients and Statistics for Age")
        print("Expected:", age)
        print("Computed:", reg.results['age'])

        ageEq = (round(age['coefficient'] - reg.results['age']['coefficient'], 1) == 0) & \
                (round(age['standard_error'] - reg.results['age']['standard_error'], 1) == 0) & \
                (round(age['z_stat'] - reg.results['age']['z_stat'], 1) == 0) & \
                (round(age['p_value'] - reg.results['age']['p_value'], 1) == 0)

        print("Testing Logit Model Coefficients and Statistics for Education")
        print("Expected:", educ)
        print("Computed:", reg.results['educ'])

        educEq = (round(educ['coefficient'] - reg.results['educ']['coefficient'], 1) == 0) & \
                 (round(educ['standard_error'] - reg.results['educ']['standard_error'], 1) == 0) & \
                 (round(educ['z_stat'] - reg.results['educ']['z_stat'], 1) == 0) & \
                 (round(educ['p_value'] - reg.results['educ']['p_value'], 1) == 0)

        print("Testing Logit Model Coefficients and Statistics for Intercept")
        print("Expected:", intercept)
        print("Computed:", reg.results['intercept'])

        interceptEq = (round(intercept['coefficient'] - reg.results['intercept']['coefficient'], 1) == 0) & \
                      (round(intercept['standard_error'] - reg.results['intercept']['standard_error'], 1) == 0) & \
                      (round(intercept['z_stat'] - reg.results['intercept']['z_stat'], 1) == 0) & \
                      (round(intercept['p_value'] - reg.results['intercept']['p_value'], 1) == 0)

        self.assertTrue((sexEq & ageEq & educEq & interceptEq), "Your coefficients are not correctly calculated.")
