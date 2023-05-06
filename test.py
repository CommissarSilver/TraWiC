from utils import santa

coder = santa.SantaCoder()

prefix = """def print_hello_wrold():
        """

suffix = """if __name__=="__main__":\n    print_hello_wrold()"""

middle = coder.infill((prefix, suffix))

print("\033[92m" + prefix + "\033[93m" + middle + "\033[92m" + suffix)
