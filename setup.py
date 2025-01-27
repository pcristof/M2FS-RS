from setuptools import setup, find_packages

setup(
    name='m2fs_rs',  # Replace with your package name
    version='0.1.0',    # Initial version
    author='Paul I. Cristofari',
    author_email='paul.ivan.cristofari@gmail.com',
    description='Package to reduce M2FS observations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',  # Replace with your repo URL
    packages=find_packages(),  # Automatically find package directories
    install_requires=[
        # Add any dependencies here, e.g., 'numpy>=1.20.0',
    ],
    python_requires='>=3.6',  # Specify the Python version
    entry_points={
        'console_scripts': [
            'm2fs.configure=m2fs_rs.helper_tools:configure',  # Command-line utility
            'm2fs.pre_reduce=scripts.pre_reduce:main',  # Command-line utility
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
