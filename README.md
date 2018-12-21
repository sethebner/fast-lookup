# fast-lookup
Fast lookup for rows in a matrix

The main file is `queue_dispatch.c`. `queue.{h|c}` contain helper functions and structs. `scripts/` contains functionality for converting text to word IDs. `text/` holds some example texts and queries.

To convert a `.txt` file into word IDs, modify the file path variables in `scripts/text2gloveids.py` to point to your files and run `python scripts/text2gloveids.py`. Similarly modify and run `scripts/token2types.py` to collapse word IDs into lists of unique IDs and the positions at which they occur.

All the hyperparameters of the lookup system are listed as `#define`s in `queue_dispatch.c` rather than as command-line arguments so that they can be compiled directly into the object code for better instruction-level optimization. After choosing some hyperparameters, run `make dispatch` to compile, and then run `./dispatch` to run the system.
