from distutils.core import setup, Extension
import numpy

module1 = Extension('spam',
                    sources = ['spammodule.c'],
                    include_dirs=[numpy.get_include()],
                    libraries=['mandel'],
                    language='c',
                    extra_compile_args = ["/openmp", "/NODEFAULTLIB:library"])

setup (name = 'SpamModule',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])