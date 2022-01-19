# -*- coding: utf-8 -*-
# Source: https://github.com/ml-uol/prosper/blob/master/prosper/utils/autotable.py
#
# For LICENSING and COPYRIGHT for this file, see prosper's license at:
# https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
#
# Author: Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#
# Modifications:
# -  Let StoreToH5.__init__ take warnings as argument

from abc import ABCMeta, abstractmethod

from os.path import isfile
from time import strftime

from mpi4py import MPI

from evo.utils.parallel import pprint
from evo.utils.autotable import AutoTable

comm = MPI.COMM_WORLD


class DataHandler(object):
    __metaclass__ = ABCMeta
    """ Base class for handler which can be set to handle incoming data by DataLog."""

    def __init__(self):
        pass

    def register(self, tblname):
        """Called by Datalog whenever this object is set as an handler for some table"""
        pass

    @abstractmethod
    def append(self, tblname, value):
        pass

    def append_all(self, valdict):
        for key, val in valdict.items():
            self.append(key, val)

    def remove(self, tblname):
        pass

    def close(self):
        pass


class StoreToH5(DataHandler):
    default_autotbl = None

    def __init__(self, destination=None, warnings=True):
        """
        Store data to the specified .h5 destination.

        *destination* may be either a file name or an existing AutoTable object
        """
        self.destination = destination

        if comm.rank == 0:
            if isinstance(destination, AutoTable):
                self.autotbl = destination
            elif isinstance(destination, str):
                self.autotbl = AutoTable(destination, warnings=warnings)
            elif destination is None:
                if StoreToH5.default_autotbl is None:
                    self.autotbl = AutoTable(warnings=warnings)
                else:
                    self.autotbl = StoreToH5.default_autotbl
            else:
                raise TypeError("Expects an AutoTable instance or a string as argument")

            if StoreToH5.default_autotbl is None:
                StoreToH5.default_autotbl = self.autotbl

    def __repr__(self):
        return "StoreToH5 into file %s" % self.destination

    def append(self, tblname, value):
        self.autotbl.append(tblname, value)

    def append_all(self, valdict):
        self.autotbl.append_all(valdict)

    def assign(self, tblname, value):
        self.autotbl.assign(tblname, value)

    def close(self):
        # if comm.rank != 0:
        # return
        self.autotbl.close()


class StoreToTxt(DataHandler):
    def __init__(self, destination=None):
        """
        Store data to the specified .txt destination.

        *destination* has to be a file name
        """
        if comm.rank == 0:
            if isinstance(destination, str):
                self.txt_file = open(destination, "w")
            elif destination is None:
                if not isfile("terminal.txt"):
                    self.txt_file = open("terminal.txt", "w")
                else:
                    raise ValueError("Please enter a file name that does not already exist.")

    def append(self, tblname, value):
        self.txt_file.write("%s = %s\n" % (tblname, value))

    def append_all(self, valdict):
        for entry in valdict.keys():
            self.txt_file.write("%s = %s\n" % (entry, valdict[entry]))

    def close(self):
        # if comm.rank != 0:
        # return
        self.txt_file.close()


class TextPrinter(DataHandler):
    def __init__(self):
        pass

    def append(self, tblname, value):
        pprint("\t%s = %s " % (tblname, value))

    def append_all(self, valdict):
        for (name, val) in valdict.items():
            pprint("\t%s = %s \n" % (name, val), end="")


class DataLog:
    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.next_vizid = 0
        self.policy = []  # Ordered list of (tbname, handler)-tuples
        self._lookup_cache = {}  # Cache for tblname -> hanlders lookups

    def _lookup(self, tblname):
        """Return a list of handlers to be used for tblname"""
        if tblname in self._lookup_cache:
            return self._lookup_cache[tblname]

        handlers = []
        for (a_tblname, a_handler) in self.policy:
            if a_tblname == tblname or a_tblname == "*":  # XXX wildcard matching XXX
                handlers.append(a_handler)
        self._lookup_cache[tblname] = handlers
        return handlers

    def progress(self, message, completed=None):
        """Append some progress message"""
        if self.comm.rank != 0:
            return

        if completed is None:
            print("[%s] %s" % (strftime("%H:%M:%S"), message))
        else:
            totlen = 65 - len(message)
            barlen = int(totlen * completed)
            spacelen = totlen - barlen
            print("[%s] %s [%s%s]" % (strftime("%H:%M:%S"), message, "*" * barlen, "-" * spacelen))

    def append(self, tblname, value):
        """Append the given value and call all the configured DataHandlers."""
        if self.comm.rank != 0:
            return

        for h in self._lookup(tblname):
            h.append(tblname, value)

    def assign(self, tblname, value):
        """Append the given value and call all the configured DataHandlers."""
        if self.comm.rank != 0:
            return

        for h in self._lookup(tblname):
            h.assign(tblname, value)

    def append_all(self, valdict):
        """
        Append the given values and call all the consigured DataHandlers

        *valdict* is expected to be a dictionary of key-value pairs.
        """
        if self.comm.rank != 0:
            return

        # Construct a set with all handlers to be called
        all_handlers = set()
        for tblname, val in valdict.items():
            hl = self._lookup(tblname)
            all_handlers = all_handlers.union(hl)

            # Call all handlers but create a personalized version
            # of valdict with oble the values this particular handler
            # is interested in
        for handler in all_handlers:
            argdict = {}
            for tblname, val in valdict.items():
                hl = self._lookup(tblname)

                if handler in hl:
                    argdict[tblname] = val

            handler.append_all(argdict)

    def ignored(self, tblname):
        """
        Returns True, then the given *name* is neither stored onto disk,
        nor visualized or triggered upon. When *ignored('something')* returns
        True, it will make no difference if you *append* a value to table *tblname* or not.

        This can be especially useful when running a (MPI-)parallel programs and collecting
        the value to be logged is an expensive operation.

        Example::

            if not dlog.ignored('summed_data'):
                summed_data =  np.empty_like(data)
                mpicomm.Reduce((data, MPI.DOUBLE), (summed_data, MPI_DOUBLE), MPI.SUM)
                dlog.append('summed_data', summed_data)

            [..]
        """
        return self._lookup(tblname) == []

    def set_handler(self, tblname, handler_class, *args, **kargs):
        """Set the specifies handler for all data stored under the name *tblname*"""
        if self.comm.rank != 0:
            return

        if not issubclass(handler_class, DataHandler):
            raise TypeError("handler_class must be a subclass of DataHandler ")

        # instantiate data handler
        handler = handler_class(*args, **kargs)
        handler.register(tblname)

        if isinstance(tblname, str):
            self.policy.append((tblname, handler))  # append to policy
        elif hasattr(tblname, "__iter__"):
            for t in tblname:
                self.policy.append((t, handler))  # append to policy
        else:
            raise TypeError("Table-name must be a string (or a list of strings)")
        return handler

    def remove_handler(self, handler):
        """Remove specified handler so that data is no longer stored there."""
        if self.comm.rank != 0:
            return

        if isinstance(handler, DataHandler):
            for a_tblname, a_handler in self.policy[:]:
                if a_handler == handler:
                    self.policy.remove((a_tblname, a_handler))
            handler.close()
            self._lookup_cache = {}
        else:
            raise ValueError("Please provide valid DataHandler object.")

    def close(self):
        """Reset the datalog and close all registered DataHandlers"""
        if self.comm.rank != 0:
            return

        for (tblname, handler) in self.policy:
            handler.close()
