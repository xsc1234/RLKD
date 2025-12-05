prompt_math = """
Decompose the problem into multiple steps according to the thinking process for reasoning. Format the thinking content in the form of [Meat-reasoning]-[Sub-problem]-[Answer] for each step. When the [Answer] of a certain step gets the answer to the original question, stop reasoning, and the last answer should include a complete answer to the original question.
[Meat-reasoning] represents the thinking process. It is necessary to analyze the previous and next steps as detailed as possible, explain in detail why this step is done to lead to the following Sub-problem, and do not mention the answer of this step.
[Sub-problem] represents the problem that needs to be solved in the current step, and analyze it as detailed as possible. Each Sub-problem contains only one question.
[Answer] represents the answer to the Sub-problem. Give the answer to the [Sub-problem] directly and briefly without any reasoning steps and redundant statements.
You should split it into more [Meat-reasoning]-[Sub-problem]-[Answer] with finer granularity, but do not repeat the conditions given in the question.

Here are some examples for math task：
Example 1:
[Question]: The duration of a process used to manufacture components is known to be normally distributed with a mean of 30 minutes and a standard deviation of 4 minutes. What is the probability of a time greater than 33 minutes being recorded?
[Response]: 
[Meat-reasoning 1]: To determine the probability that the process time exceeds 33 minutes, we first need to standardize the value using the z-score formula. This converts the normal distribution parameters (mean=30, standard deviation=4) into the standard normal distribution (mean=0, standard deviation=1), allowing us to use standard tables or tools for probability lookup.-  
[Sub-problem 1]: What is the z-score for a time of 33 minutes in this normal distribution?  
[Answer 1]: 0.75  

[Meat-reasoning 2]: With the z-score calculated, the next step is to find the area under the standard normal curve to the right of this z-score, which corresponds to the probability of the process time being greater than 33 minutes. Since standard tables typically provide cumulative probabilities (area to the left), subtracting the cumulative probability from 1 gives the desired result.-  
[Sub-problem 2]: What is the cumulative probability (area to the left) for a z-score of 0.75 in the standard normal distribution?  
[Answer 2]: 0.7734  

[Meat-reasoning 3]: Finally, subtract the cumulative probability from 1 to obtain the probability of the time exceeding 33 minutes. This completes the calculation.-  
[Sub-problem 3]: What is 1 minus the cumulative probability of 0.7734?  
[Answer 3]: 0.2266 
So the final answer is:
-0.2266.

Example 2:
[Question]: Given real numbers \\( a, b, c \\) and a positive number \\( \\lambda \\) such that the polynomial \\( f(x) = x^3 + a x^2 + b x + c \\) has three real roots \\( x_1, x_2, x_3 \\), and the conditions \\( x_2 - x_1 = \\lambda \\) and \\( x_3 > \\frac{1}{2}(x_1 + x_2) \\) are satisfied, find the maximum value of \\( \\frac{2 a^3 + 27 c - 9 a b}{\\lambda^3} \\).
[Response]: 
[Meat-reasoning 1]: To find the maximum value of \( \frac{2a^3 + 27c - 9ab}{\lambda^3} \), use Vieta’s formulas to express coefficients \( a, b, c \) in terms of the roots \( x_1, x_2, x_3 \). Parametrize the roots to satisfy \( x_2 - x_1 = \lambda \) and \( x_3 > \frac{1}{2}(x_1 + x_2) \). Define \( x_1 = t - \frac{\lambda}{2} \), \( x_2 = t + \frac{\lambda}{2} \), and \( x_3 = t + s \), where \( s > 0 \). This parametrization simplifies the problem by eliminating \( t \) through symmetry.-  
[Sub-problem 1]: Express \( a, b, c \) in terms of \( t \), \( s \), and \( \lambda \) using the parametrized roots.-  
[Answer 1]: \( a = -3t - s \), \( b = 3t^2 + 2ts - \frac{\lambda^2}{4} \), \( c = -t^3 - t^2s + \frac{\lambda^2}{4}(t + s) \).  

[Meat-reasoning 2]: Substitute \( a, b, c \) into the expression \( \frac{2a^3 + 27c - 9ab}{\lambda^3} \). Simplify the expression by combining like terms. The result will depend only on \( s \) and \( \lambda \), allowing further simplification by introducing \( k = \frac{s}{\lambda} \). This reduces the problem to maximizing a cubic function in \( k \).-  
[Sub-problem 2]: Simplify \( \frac{2a^3 + 27c - 9ab}{\lambda^3} \) after substitution. What is the simplified form in terms of \( k \)?  
[Answer 2]: \( -2k^3 + \frac{9}{2}k \).  

[Meat-reasoning 3]: To maximize \( Q(k) = -2k^3 + \frac{9}{2}k \), find its critical points by taking the derivative and solving for \( k \). Verify that the critical point corresponds to a maximum using the second derivative test.-  
[Sub-problem 3]: Find the critical point of \( Q(k) = -2k^3 + \frac{9}{2}k \).  
[Answer 3]: \( k = \frac{\sqrt{3}}{2} \).  

[Meat-reasoning 4]: Substitute \( k = \frac{\sqrt{3}}{2} \) back into \( Q(k) \) to compute the maximum value. Confirm the calculation to ensure no arithmetic errors.-  
[Sub-problem 4]: Calculate \( Q\left(\frac{\sqrt{3}}{2}\right) \).  
[Answer 4]: \( \frac{3\sqrt{3}}{2} \).  

[Meat-reasoning 5]: Verify that the critical point \( k = \frac{\sqrt{3}}{2} \) satisfies all constraints (e.g., \( s > 0 \)) and that the polynomial discriminant condition for three real roots is implicitly satisfied by the problem’s given conditions. Confirm no additional constraints affect the result.-  
[Sub-problem 5]: Does \( k = \frac{\sqrt{3}}{2} \) satisfy \( s > 0 \)?  
[Answer 5]: Yes.  

[Meat-reasoning 6]: Finalize the maximum value of the original expression based on the derived result.-  
[Sub-problem 6]: What is the maximum value of \( \frac{2a^3 + 27c - 9ab}{\lambda^3} \)?  
[Answer 6]: \( \frac{3\sqrt{3}}{2} \).

So the final answer is:
\( \frac{3\sqrt{3}}{2} \).

[Question]: 
"""

prompt_code = """
Decompose the problem into multiple steps according to the thinking process for reasoning. Format the thinking content in the form of [Meat-reasoning]-[Sub-problem]-[Answer] for each step. When the [Answer] of a certain step gets the answer to the original question, stop reasoning, and the last answer should include a complete answer to the original question.
[Meat-reasoning] represents the thinking process. It is necessary to analyze the previous and next steps as detailed as possible, explain in detail why this step is done to lead to the following Sub-problem, and do not mention the answer of this step.
[Sub-problem] represents the problem that needs to be solved in the current step, and analyze it as detailed as possible. Each Sub-problem contains only one question.
[Answer] represents the answer to the Sub-problem. Give the answer to the [Sub-problem] directly and briefly without any reasoning steps and redundant statements.
You should split it into more [Meat-reasoning]-[Sub-problem]-[Answer] with finer granularity, but do not repeat the conditions given in the question.

Here are some examples for coding task：
Example 1:
[Question]: Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition.Vanya got an important task \u2014 he should enumerate books in the library and label each book with its number. Each of the n books should be assigned with a number from 1 to n. Naturally, distinct books should be assigned distinct numbers.\n\nVanya wants to know how many digits he will have to write down as he labels the books.\n\n\n-----Input-----\n\nThe first line contains integer n (1 \u2264 n \u2264 10^9) \u2014 the number of books in the library.\n\n\n-----Output-----\n\nPrint the number of digits needed to number all the books.\n\n\n-----Examples-----\nInput\n13\n\nOutput\n17\n\nInput\n4\n\nOutput\n4\n\n\n\n-----Note-----\n\nNote to the first test. The books get numbers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, which totals to 17 digits.\n\nNote to the second sample. The books get numbers 1, 2, 3, 4, which totals to 4 digits.\n
[Response]: 
[Meat-reasoning 1]: The problem requires calculating the total number of digits written when labeling books from 1 to n. The approach involves iterating through each digit length (1-digit, 2-digits, etc.) and summing the contributions of each range. For example, numbers with d digits start at 10^(d-1) and end at 10^d -1. If n is within this range, calculate the count of numbers in the partial range.-
[Sub-problem 1]: How to determine the contribution of numbers with a specific digit length d to the total digit count?
[Answer 1]: For each d, calculate the start and end of the range. If n >= end, add (end - start +1) * d. If n < end, add (n - start +1) * d.

[Meat-reasoning 2]: Implementing this requires iterating through each possible digit length until the start of the range exceeds n. The loop increments d and calculates the start and end for each iteration. The total accumulates the contributions from each valid range.-
[Sub-problem 2]: How to structure the loop to iterate through digit lengths and accumulate the total digits?
[Answer 2]: Initialize total=0, d=1. Loop while start <=n, update total based on current d's range, increment d.

[Meat-reasoning 3]: The code must read n from standard input, compute the total digits, and print the result. Handling edge cases like n=1 or n=1e9 correctly is ensured by checking the range boundaries.-
[Sub-problem 3]: What is the Python function structure to read input, compute the total digits, and print the output?
[Answer 3]: Read n with int(input()), loop through d starting at 1, calculate start and end, update total, then print(total).

[Meat-reasoning 4]: Finally, the function must be called after definition to execute when the script runs. The loop terminates when start exceeds n, ensuring all relevant digit lengths are covered.-
[Sub-problem 4]: What is the complete Python function implementing the described logic?
[Answer 4]:
def count_digits():
n = int(input())
total = 0
d = 1
while True:
start = 10 ** (d-1)
if start > n:
break
end = 10**d -1
count = end - start +1 if end <=n else n - start +1
total += count * d
d +=1
print(total)
count_digits()

So the final answer is:
def count_digits():
n = int(input())
total = 0
d = 1
while True:
start = 10 ** (d-1)
if start > n:
break
end = 10**d -1
count = end - start +1 if end <=n else n - start +1
total += count * d
d +=1
print(total)
count_digits()

Example 2:
[Question]: Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition.You are given n days and for each day (d_{i}) you can select one of the following options:\n\tperform no task \n\tperform a high effort task (h_{i}) only if its the first day or if you chose no-task on the previous day\n\tperform a low effort task (l_{i})\nWrite a program to find the maximum amount of tasks you can perform within these n days. \nExample 1:\nInput:\nn = 3\nhi[] = {2,8,1}\nli[] = {1,2,1}\nOutput: 9\nExplanation:\nOptions on 1st day: hi[0]=2, li[0]=1 or no-task\nSelect no-task.\nOptions on 2nd day: hi[1]=8, li[2]=1 or no-task\nSelect high-effort task as no-task was selected \nthe previous day.  \nOptions on 3rd day: hi[2]=1, li[2]=1 or no-task\nSelect low-effort task as selecting high-effort \ntask is only possible if you chose no-task on \nprevious day.\nExample 2:\nInput:\nn = 5\nhi[] = {3,6,8,7,6}\nli[] = {1,5,4,5,3}\nOutput: 20\nExplanation: Perform high-effort task on first \nday and low-effort task on all remaining days.\n \nYour Task:\nYou don't need to read input or print anything. Your task is to complete the function maxAmt() which accepts an integer n and two arrays li[] and hi[] as input parameter and returns the maximum amount of tasks you can perform within these n days.\nExpected Time Complexity: O(n)\nExpected Auxiliary Space: O(n)\nConstraints:\n1 <= n <= 10^{6}\n#User function Template for python3\n\nclass Solution:\n    def maxAmt(self, n, hi, li): \n        # code here \n
[Response]: 
[Meat-reasoning 1]: To maximize the total task amount with constraints on high-effort tasks, dynamic programming is suitable. We need to track the maximum possible sum for each state (no-task, high, low) at each day, considering transitions based on previous choices.-  
[Sub-problem 1]: What are the possible states for each day and their transitions?  
[Answer 1]: States: no-task (0), high (1), low (2). Transitions: high can follow only no-task; others depend on max previous states.  

[Meat-reasoning 2]: For each day, the current state's value depends on the previous day's states. For example, choosing high on day i requires the previous day to be no-task. We need to define recurrence relations for dynamic programming.-  
[Sub-problem 2]: What are the recurrence relations for each state?  
[Answer 2]: dp[i][0] = max(prev0, prev1, prev2); dp[i][1] = prev0 + hi[i]; dp[i][2] = max_prev + li[i].  

[Meat-reasoning 3]: To optimize space, track only the previous day's states instead of a full DP array. This reduces space complexity to O(1) while maintaining O(n) time.-  
[Sub-problem 3]: How to implement the transitions using variables instead of a DP array?  
[Answer 3]: Use variables prev0, prev1, prev2 and update them iteratively.  

[Meat-reasoning 4]: Initialize the states for day 0, then iterate from day 1 to n-1. After processing all days, the answer is the maximum of the final states.-  
[Sub-problem 4]: What is the Python function implementing this logic?  
[Answer 4]: Function uses a loop to update curr0, curr1, curr2 based on previous states and returns their maximum.  

```python
class Solution:
    def maxAmt(self, n, hi, li):
        if n == 0:
            return 0
        prev0, prev1, prev2 = 0, hi[0], li[0]
        for i in range(1, n):
            curr0 = max(prev0, prev1, prev2)
            curr1 = prev0 + hi[i]
            curr2 = curr0 + li[i]
            prev0, prev1, prev2 = curr0, curr1, curr2
        return max(prev0, prev1, prev2)


So the final answer is:
```python
class Solution:
    def maxAmt(self, n, hi, li):
        if n == 0:
            return 0
        prev0, prev1, prev2 = 0, hi[0], li[0]
        for i in range(1, n):
            curr0 = max(prev0, prev1, prev2)
            curr1 = prev0 + hi[i]
            curr2 = curr0 + li[i]
            prev0, prev1, prev2 = curr0, curr1, curr2
        return max(prev0, prev1, prev2)

[Question]: 
"""

prompt_science = """
Decompose the problem into multiple steps according to the thinking process for reasoning. Format the thinking content in the form of [Meat-reasoning]-[Sub-problem]-[Answer] for each step. When the [Answer] of a certain step gets the answer to the original question, stop reasoning, and the last answer should include a complete answer to the original question.
[Meat-reasoning] represents the thinking process. It is necessary to analyze the previous and next steps as detailed as possible, explain in detail why this step is done to lead to the following Sub-problem, and do not mention the answer of this step.
[Sub-problem] represents the problem that needs to be solved in the current step, and analyze it as detailed as possible. Each Sub-problem contains only one question.
[Answer] represents the answer to the Sub-problem. Give the answer to the [Sub-problem] directly and briefly without any reasoning steps and redundant statements.
You should split it into more [Meat-reasoning]-[Sub-problem]-[Answer] with finer granularity, but do not repeat the conditions given in the question.

Here are some examples for scientific task：
Example 1:
[Question]: What is the concentration (in mol\/L) of a hydrochloric acid (HCl) solution if 25 mL of this solution requires 35 mL of a 0.1 M sodium hydroxide (NaOH) solution to reach the equivalence point in a titration reaction?
[Response]: 
[Meat-reasoning 1]: To determine the concentration of HCl, the first step is to write the balanced chemical equation for the reaction between HCl and NaOH. This establishes the stoichiometric relationship between the reactants, which is necessary for subsequent calculations.-
[Sub-problem 1]: What is the balanced chemical equation for the reaction of HCl with NaOH, and what is the mole ratio between them?
[Answer 1]: HCl + NaOH → NaCl + H₂O (1:1 mole ratio)

[Meat-reasoning 2]: Since the mole ratio is 1:1, the moles of HCl will equal the moles of NaOH at equivalence. The moles of NaOH can be calculated using its concentration and volume, which will directly give the moles of HCl.-
[Sub-problem 2]: How many moles of NaOH are present in 35 mL of a 0.1 M solution?
[Answer 2]: 0.0035 mol

[Meat-reasoning 3]: With the moles of HCl determined (equal to moles of NaOH), the concentration of the HCl solution is calculated by dividing the moles of HCl by its volume in liters. This requires converting the given HCl volume from mL to L.-
[Sub-problem 3]: What is the volume of the HCl solution in liters?
[Answer 3]: 0.025 L

[Meat-reasoning 4]: Finally, divide the moles of HCl by its volume in liters to obtain the concentration. This completes the calculation.-
[Sub-problem 4]: What is the concentration (mol/L) of HCl when 0.0035 mol is dissolved in 0.025 L?
[Answer 4]: 0.14 mol/L

So the final answer is:
0.14 mol/L

Example 2:
[Question]: A circular loop of radius 0.2 meters carries a current of 5 amperes. What is the magnetic field at a point located 0.1 meters from the center of the loop?
[Response]: 
[Meat-reasoning 1]: To determine the magnetic field at a point 0.1 meters from the center of a circular current-carrying loop, we first identify the relevant formula. Since the point's location is not explicitly stated as off-axis, we assume it lies along the loop's axis. The formula for the magnetic field on the axis of a circular loop is required here.  
[Sub-problem 1]: What is the formula for the magnetic field at a point on the axis of a circular current-carrying loop?  
[Answer 1]: B = (μ₀ * I * R²) / [2*(R² + z²)^(3/2)]  

[Meat-reasoning 2]: Next, substitute the given values into the formula: current (I = 5 A), loop radius (R = 0.2 m), axial distance (z = 0.1 m), and permeability of free space (μ₀ = 4π × 10⁻⁷ T·m/A). Compute each component stepwise to avoid errors.  
[Sub-problem 2]: What is the value of R² + z² for R = 0.2 m and z = 0.1 m?  
[Answer 2]: 0.05 m²  

[Meat-reasoning 3]: Calculate the term (R² + z²) raised to the power of 3/2, which appears in the denominator of the formula. This involves first taking the square root and then cubing the result. 
[Sub-problem 3]: What is (0.05 m²) raised to the 3/2 power?  
[Answer 3]: 0.01118 m³  

[Meat-reasoning 4]: Now compute the numerator of the formula: μ₀ * I * R². Use μ₀ = 4π × 10⁻⁷ T·m/A, I = 5 A, and R² = 0.04 m².  
[Sub-problem 4]: What is the value of μ₀ * I * R²?  
[Answer 4]: 2.5133 × 10⁻⁷ T·m³  

[Meat-reasoning 5]: Finally, divide the numerator by the denominator (2 * 0.01118 m³) to obtain the magnetic field strength at the specified point.-  
[Sub-problem 5]: What is 2.5133 × 10⁻⁷ T·m³ divided by 0.02236 m³?  
[Answer 5]: 1.1237 × 10⁻⁵ T

So the final answer is:
1.1237 × 10⁻⁵ T

[Question]: 
"""

prompt_puzzle = """
Decompose the problem into multiple steps according to the thinking process for reasoning. Format the thinking content in the form of [Meat-reasoning]-[Sub-problem]-[Answer] for each step. When the [Answer] of a certain step gets the answer to the original question, stop reasoning, and the last answer should include a complete answer to the original question.
[Meat-reasoning] represents the thinking process. It is necessary to analyze the previous and next steps as detailed as possible, explain in detail why this step is done to lead to the following Sub-problem, and do not mention the answer of this step.
[Sub-problem] represents the problem that needs to be solved in the current step, and analyze it as detailed as possible. Each Sub-problem contains only one question.
[Answer] represents the answer to the Sub-problem. Give the answer to the [Sub-problem] directly and briefly without any reasoning steps and redundant statements.
You should split it into more [Meat-reasoning]-[Sub-problem]-[Answer] with finer granularity, but do not repeat the conditions given in the question.

Here are some examples for puzzle task：
Example 1:
[Question]: Where might  jellyfish stay warm?\nA: atlantic ocean\nB: bikini bottom\nC: japanese restaurant\nD: store\nE: pacific ocean
[Response]: 
[Meat-reasoning 1]: To determine where jellyfish might stay warm, first identify their natural habitats. Jellyfish are marine creatures, so options related to oceans or aquatic environments are primary candidates. This narrows the focus to real-world oceans (A, E) or fictional underwater locations (B).
[Sub-problem 1]: Are jellyfish typically found in real oceans or fictional locations like Bikini Bottom?
[Answer 1]: Real oceans

[Meat-reasoning 2]: Since jellyfish inhabit real oceans, eliminate fictional or non-marine options. The remaining choices are Atlantic Ocean (A) and Pacific Ocean (E). However, the question specifies "stay warm," implying a need to consider temperature variations within these oceans.
[Sub-problem 2]: Do both the Atlantic and Pacific Oceans have warm regions suitable for jellyfish?
[Answer 2]: Yes

[Meat-reasoning 3]: Since both A and E are valid, re-examine the options for potential context clues. The inclusion of "Bikini Bottom" (a fictional warm underwater setting from SpongeBob SquarePants) suggests the question might reference pop culture rather than biology.
[Sub-problem 3]: Is "Bikini Bottom" a known fictional habitat for jellyfish in media?
[Answer 3]: Yes

[Meat-reasoning 4]: Given the mix of real and fictional options, infer the question’s intent: if it prioritizes humor or pop culture, Bikini Bottom (B) is likely the answer. If strictly biological, A/E would apply. However, the presence of B suggests a non-literal context.
[Sub-problem 4]: Which option aligns with a playful or media-reference interpretation of "stay warm"?
[Answer 4]: B: Bikini Bottom

So the final answer is:
B: Bikini Bottom

Example 2:
[Question]: As you live life you get wiser, it is a part of what?\nA: growing older\nB: death\nC: experience things\nD: toothsome\nE: breathe
[Response]: 
[Meat-reasoning 1]: The question asks to identify what gaining wisdom through living life is a part of. First, analyze the options to eliminate unrelated terms (e.g., "toothsome" and "breathe" are irrelevant to wisdom). Then, focus on comparing plausible options like "growing older" and "experience things."-  
[Sub-problem 1]: Which options directly relate to the process of accumulating wisdom as stated in the question?  
[Answer 1]: A: growing older; C: experience things  

[Meat-reasoning 2]: To differentiate between "growing older" and "experience things," evaluate whether wisdom arises purely from aging or from actively engaging with life. Wisdom typically stems from learning through actions and events rather than mere passage of time.-  
[Sub-problem 2]: Is gaining wisdom more closely tied to aging or to encountering and reflecting on life events?  
[Answer 2]: C: experience things  

[Meat-reasoning 3]: Confirm that "experience things" logically encompasses the idea of learning and gaining wisdom through lived events, making it the most fitting answer.-  
[Sub-problem 3]: Which option explicitly connects wisdom to lived actions rather than passive aging?  
[Answer 3]: C: experience things

So the final answer is:
C: experience things

[Question]: 
"""