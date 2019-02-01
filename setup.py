from distutils.core import setup, Extension
import numpy

module1 = Extension('spam',
                    sources = ['spammodule.c'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args = [])

setup (name = 'SpamModule',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
