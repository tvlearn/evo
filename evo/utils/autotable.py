# -*- coding: utf-8 -*-
# Source: https://github.com/ml-uol/prosper/blob/master/prosper/utils/autotable.py
#
# For LICENSING and COPYRIGHT for this file, see prosper's license at:
# https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
#
# Modifications:
# -  Let AutoTable.__init__ take warnings as argument

"""
The autotable module provides a simple interface to store data from simulation
runs into efficient HDF5 files onto the filesystem.

These files can later be opened

 * in pure C or C++: libhdf5 (http://www.hdfgroup.org/HDF5/)
 * in Python: PyTables (http://www.pytables.org/) or h5py (http://www.h5py.org/)
 * in Matlab: hdf5read( <filename>, <tablename> )

Basic example::

    import numpy as np
    from pulp.utils import autotable

    tbl = autotable.AutoTable('~/testhdf.h5')
    for t in range(10):
        tbl.append('t', t)
        tbl.append('D', np.random.randn(20))
    tbl.close()

This code creates the file :file:`~/testhdf.h5` with two tables, each having 10
rows: The table *t* will store a single integer within each row, where the
table *D* will store a 20 element vector with gaussian distributed random
numbers in each row.
"""

import numpy as np
import tables


class AutoTable:
    """Store data into HDF5 files"""

    def __init__(self, fname=None, compression_level=1, rwmode="w", warnings=True):
        """Create a new autotable object which will write data into a file called fname.

        :param fname: Filename for the HDF5 file to be created. (default: None)
        :type  fname: string
        :param compression_level: Specifies the compression level used when storing data.
                                  (default: 1)
        :type  compression_level: int

        If fname is not specified (or is None), fname will be derived from sys.argv[0] by striping
        the extension and adding ".h5".  As a result, the file will be named like the creating
        program.

        This class implements the context-manager interface such that

        >>> with AutoTable('test.h5') as tbl:
        >>>     tbl.append('t', 0)

        correctly opens and closes the HDF file even in the case of exceptions.


        .. note:: If a file named fname existed previously, its content will be deleted!
        """
        self.warnings = warnings
        if fname is None:
            fname = self._guess_fname()
        self.h5 = tables.open_file(fname, rwmode)
        self.compression_level = compression_level
        self.tables = {}
        self.types = {}

    def __enter__(self):
        """Behave like a proper Python context manager

        Example

        >>> with AutoTable("test.h5") as tbl:
        >>>     tbl.append('a', 10)
        """
        return self

    def __exit__(self, *exc_info):
        """Behave like a proper Python context manager"""
        self.close()

    def close(self):
        """Close the HDF file behind this AutoTable instance."""
        self.h5.close()

    def append(self, name, value):
        """Append the dataset *values* into a table called *name*.

        :param name: Name of the table to save into.
        :type  name: strring
        :param value: Data to be stored.
        :type  value: int or float or ndarray

        If a specified table name does not exist, a new table will be created.

        Examples

        >>> tbl.append("T", temp)
        >>> tbl.append("image", np.zeros((256,256)) )
        """
        if type(value) == np.ma.core.MaskedArray:
            value = value.data

        if type(value) == str:
            self._appendstr(name, value)
            return

        if np.isscalar(value):
            value = np.asarray(value)

        if not isinstance(value, np.ndarray):
            raise TypeError("Don't know how to handle values of type '%s'", type(value))

        # Check if we need to create a new table
        if name not in self.tables:
            self._create_table(name, value)

        value = value.reshape((1,) + value.shape)
        try:
            self.tables[name].append(value)
        except ValueError:
            raise TypeError('Wrong datatype "%s" for "%s" field' % (value.dtype, name))

        self.tables[name].flush()

    def assign(self, name, value):
        """Assigns the dataset *values* into a table called *name*.

        If the specified table exist beforehand, the old data will be overwritten.

        :param name: Name of the table to save into.
        :type  name: strring
        :param value: Data to be stored.
        :type  value: int or float or ndarray


        Examples

        >>> tbl.assign("T", temp)
        >>> tbl.assign("image", np.zeros((256,256)) )
        """
        if type(value) == str:
            self._appendstr(name, value)
            return

        if np.isscalar(value):
            value = np.asarray(value)
            value = value.reshape((1,) + value.shape)

        if not isinstance(value, np.ndarray):
            raise TypeError("Don't know how to handle values of type '%s'", type(value))

        if name not in self.tables:
            pass
        else:
            if self.warnings:
                print("Warning! The previous data with key %s is being overwritten" % name)
            self._delete_table(name)

        try:
            for ii in range(value.shape[0]):
                self.append(name, value[ii])
        except ValueError:
            raise TypeError('Wrong datatype for "%s" field' % name)

        self.tables[name].flush()

    def append_all(self, valdict):
        """Append all the key-value pairs to the table.

        *valdict* must be a dictionary containig key value pairs, where key
        is a string and specifies the table name. The corresponding value must be
        an arbitrary numpy compatible value. If a specified table name does not
        exist, a a new table will be created.

        :param valdict: Key/value pairs to be saved.
        :type  valdict: dict
        """
        for name, value in valdict.items():
            self.append(name, value)

    def appendList(self, name, value):
        """Append the dataset *values* into a table called *name*.

        Different from "append", it will add a list of entry into the table. (value, instead
        of containing one entry, contains a list of entries.)
        If a specified table name does not exist, a a new table will be created.

        Examples

        >>> tbl.append("T", temp)
        >>> tbl.append("image", np.zeros((256,256)) )
        """
        if type(value) == list and type(value[0]) == str:
            self._appendstrList(name, value)
            return

        if np.isscalar(value):
            value = np.asarray(value)

        if not isinstance(value, np.ndarray):
            raise TypeError("Don't know how to handle values of type '%s'", type(value))

        # Check if we need to create a new table
        if name not in self.tables:
            self._create_table_list(name, value)

        value = value.reshape(value.shape)
        try:
            self.tables[name].append(value)
        except ValueError:
            raise TypeError('Wrong datatype for "%s" field' % name)

        self.tables[name].flush()

    def _delete_table(self, name):
        """
        Delete a node from the h5-table together with all dictionary entries
        that has been created with the node.
        """
        self.h5.remove_node("/", name)
        del self.tables[name]
        del self.types[name]

    def _create_table(self, name, example):
        """
        Create a new table within the HDF file, where the tables shape and its
        datatype are determined by *example*.
        """
        type_map = {
            np.dtype(np.float64): tables.Float64Atom(),
            np.dtype(np.float32): tables.Float32Atom(),
            np.dtype(np.int64): tables.Int64Atom(),
            np.dtype(np.int8): tables.Int8Atom(),
            np.dtype(np.uint8): tables.UInt8Atom(),
            np.dtype(np.int16): tables.Int16Atom(),
            np.dtype(np.uint16): tables.UInt16Atom(),
            np.dtype(np.int32): tables.Int32Atom(),
            np.dtype(np.uint32): tables.UInt32Atom(),
            np.dtype(np.bool_): tables.BoolAtom(),
        }

        try:
            if type(example) == np.ndarray:
                h5type = type_map[example.dtype]
            elif type(example) == str:
                h5type = tables.VLStringAtom()
        except KeyError:
            raise TypeError(
                "Could not create table %s because of unknown dtype '%s'" % (name, example.dtype)
            )  # + ", of name: " % example.shape)

        if type(example) == np.ndarray:
            h5dim = (0,) + example.shape

            h5 = self.h5
            filters = tables.Filters(complevel=self.compression_level, complib="zlib", shuffle=True)
            self.tables[name] = h5.create_earray(h5.root, name, h5type, h5dim, filters=filters)
        elif type(example) == str:
            h5 = self.h5
            filters = tables.Filters(complevel=self.compression_level, complib="zlib", shuffle=True)
            self.tables[name] = h5.create_vlarray(h5.root, name, h5type, filters=filters)
        self.types[name] = type(example)

    def _create_table_list(self, name, example):
        """
        Create a new table within the HDF file, where the tables shape and its
        datatype are determined by *example*.
        The modified version for creating table with appendList
        """
        type_map = {
            np.dtype(np.float64): tables.Float64Atom(),
            np.dtype(np.float32): tables.Float32Atom(),
            np.dtype(np.int64): tables.Int64Atom(),
            np.dtype(np.int8): tables.Int8Atom(),
            np.dtype(np.uint8): tables.UInt8Atom(),
            np.dtype(np.int16): tables.Int16Atom(),
            np.dtype(np.uint16): tables.UInt16Atom(),
            np.dtype(np.int32): tables.Int32Atom(),
            np.dtype(np.uint32): tables.UInt32Atom(),
            np.dtype(np.bool_): tables.BoolAtom(),
        }

        try:
            if type(example) == np.ndarray:
                h5type = type_map[example.dtype]
            elif type(example) == list and type(example[0]) == str:
                h5type = tables.VLStringAtom()
        except KeyError:
            raise TypeError("Don't know how to handle dtype '%s'" % example.dtype)

        if type(example) == np.ndarray:
            h5dim = (0,) + example.shape[1:]

            h5 = self.h5
            filters = tables.Filters(complevel=self.compression_level, complib="zlib", shuffle=True)
            self.tables[name] = h5.create_earray(h5.root, name, h5type, h5dim, filters=filters)
        elif type(example) == list and type(example[0]) == str:
            h5 = self.h5
            filters = tables.Filters(complevel=self.compression_level, complib="zlib", shuffle=True)
            self.tables[name] = h5.create_vlarray(h5.root, name, h5type, filters=filters)
        self.types[name] = type(example)

    def _guess_fname(self):
        """
        Derive an fname from sys.argv[0] by striping the extension and adding ".h5".
        As a result, the table will be named just like the executing programm.
        """
        import sys
        import os.path as path

        base, _ = path.splitext(sys.argv[0])
        return base + ".h5"

    def _appendstr(self, name, value):
        """
        Removing the numpy specific operation in appending
        """
        # Check if we need to create a new table
        if name not in self.tables:
            self._create_table(name, value)

        try:
            self.tables[name].append(value)
        except ValueError:
            raise TypeError(
                "Could not create table %s because of unknown dtype '%s'" % (name, value.dtype)
            )
            # raise TypeError('Wrong datatype for "%s" field'%name)

        self.tables[name].flush()

    def _appendstrList(self, name, value):
        """
        Removing the numpy specific operation in appending
        """
        # Check if we need to create a new table
        if name not in self.tables:
            self._create_table_list(name, value)

        try:
            map(self.tables[name].append, value)
        except ValueError:
            raise TypeError('Wrong datatype for "%s" field' % name)

        self.tables[name].flush()
