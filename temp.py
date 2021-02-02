def computeGCD(x, y):
    while (y):
        x, y = y, x % y

    return x

a = 2
b = 6
print ("The gcd of 60 and 48 is : ",end="")
print (computeGCD(60,48))


def gcd(a, b):
    if a == 0:
        return b
    return gcd(b % a, a)


# Function to return LCM of two numbers
def lcm(a, b):
    return (a / gcd(a, b)) * b


# Driver program to test above function
a = 24
b = 36
print('LCM of', a, 'and', b, 'is', lcm(a, b))