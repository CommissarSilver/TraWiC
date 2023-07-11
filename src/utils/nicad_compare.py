import os

# run this command in the directory where the NiCad clone is located
print(f"{os.path.join(os.getcwd(),'NiCad-6.2')}/nicad6 functions py {os.path.join(os.getcwd(),'NiCad-6.2','systems','dd')} default-report")
os.system(
    f"{os.path.join(os.getcwd(),'NiCad-6.2')}/nicad6 functions py {os.path.join(os.getcwd(),'NiCad-6.2','systems','dd')} default-report"
)
