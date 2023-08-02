from scipy.fft import fft, ifft

"""
    function which uses the scipy.fft and scipy.ifft to calculate product of 2 large numbers
"""
def getProduct(x,y):
    xlen = len(str(x))
    ylen = len(str(y))

    #storing the digits in each number in the array
    x1 = []
    y1 = []

    tx = x
    ty = y

    # To get the digits of number in polynomial form
    for _ in range(xlen+ylen):
        x1.append(tx%10)
        tx = tx // 10
        y1.append(ty%10)
        ty = ty//10
    
    # computing the fft of each of the polynomial
    xfft = fft(x1)
    yfft = fft(y1)

    # multiplying ffts and computing back its inverse
    product = xfft*yfft
    product = ifft(product)
    ans = 0

    # computing the final ans from the array of product by multiplying with appropriate powers of 10
    for i in range(len(product)):
        ans += product[i].real*(10**i)
    
    return ans


if __name__ == "__main__":
    a = 9322356
    b = 8922002
    calcAns = getProduct(a, b)
    actualAns = a*b
    print("Actual Product: {}".format(actualAns))
    print("Product calculated using fft: {}\n".format(calcAns))