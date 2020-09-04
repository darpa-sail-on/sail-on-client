CLI Interface
-------------

To run experiments on different tests, Sail-On client relies on a command line
interface(CLI) to provide a json based configuration with the appropriate value for
different components.

The CLI supports provides the following options::

  protocol_file: path to a python file containing the empirical protocol
  -p protocol_config: path to a json based config file containing parameters for the experiment
  -a, algorithms: root of the algorithms directory (optional)
  -g, --generate: Generate template algorithm files
  -i, --interface:  Name of the Interface class that would be used for communication
  -l, --list_interfaces: Print the list of available interfaces for a protocol
