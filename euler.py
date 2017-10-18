import getopt
import sys


def is_palindrome(n):
    return str(n) == str(n)[::-1]


def largest_palindrome(n_digits=3):
    biggest = 0
    for i in range((10 ** n_digits - 1), (10 ** (n_digits - 1)) - 1, -1):
        for j in range((10 ** n_digits - 1), (10 ** (n_digits - 1)) - 1, -1):
            product = j * i
            if product > biggest and is_palindrome(product):
                biggest = product

    return biggest


def prime_factorization(n):
    i = 2
    primes = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            primes.append(i)

    if n > 1:
        primes.append(n)

    return primes


def largest_prime_factor(n=600851475143):
    return max(prime_factorization(n))


def fibonacci(n):
    if n == 0:
        raise ValueError('Invalid Argument: UNDEFINED for 0')
    elif n < 3:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibo_sum_even(upper=4000000):
    n = 0
    i = 2
    total = 0
    while True:
        n = fibonacci(i)
        if n < upper:
            i += 3
            total += n
        else:
            break

    return total


def three_five_multiples(below=1000):
    total = 0
    for i in range(below):
        if not i % 3 or not i % 5:
            total += i

    return total


def process(arg):
    if arg is '1':
        print three_five_multiples()
    elif arg is '2':
        print fibo_sum_even()
    elif arg is '3':
        print largest_prime_factor()
    elif arg is '4':
        print largest_palindrome()
    else:
        raise ValueError('Not Implemented')


def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    except getopt.error, msg:
        print msg
        print 'for help use --help'
        sys.exit(2)
    # process options
    for o in opts:
        if o in ('-h', '--help'):
            print __doc__
            sys.exit(0)

    # process arguments
    for arg in args:
        process(arg)


if __name__ == '__main__':
    sys.exit(main())
