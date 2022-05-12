Configurable Components
=======================

Providing configurable abstractions requires two steps

1. Parsing 1 or more configuration to extract the appropriate configuration values
   from it
2. Using the values to instantiate the object

We use `Hydra`_  for (1) and `SMQTK`_ for (2). Hydra uses YAML based hierarchical
configurations. The hierarchy is defined under the `configs`_ folder and provides
defaults for the abstractions and features provided by sail-on-client

.. code-block::
   :caption: Configuration hierarchy present in sail-on-client
  .
  ├── default.yaml
  ├── base
  │   ├── base_ond.yaml (default)
  │   └── base_condda.yaml
  ├── harness
  │   ├── local.yaml (default)
  │   └── par.yaml
  ├── protocol
  │   ├── ond.yaml (default)
  │   ├── condda.yaml
  │   ├── visual.yaml
  │   ├── detection
  │   │   ├── system.yaml (default)
  │   │   └── given.yaml
  │   ├── eval
  │   │   ├── none.yaml (default)
  │   │   ├── without_reaction.yaml
  │   │   └── with_reaction.yaml
  │   ├── feedback
  │   │   ├── none.yaml (default)
  │   │   ├── classification.yaml
  │   │   ├── detection_and_classification.yaml
  │   │   ├── detection.yaml
  │   │   ├── label.yaml
  │   │   └── score.yaml
  │   ├── resume_session
  │   │   ├── none.yaml (default)
  │   │   └── resume.yaml
  │   ├── save_attributes
  │   │   ├── none.yaml (default)
  │   │   └── save_attributes.yaml
  │   ├── save_features
  │   │   ├── none.yaml (default)
  │   │   ├── elementwise.yaml
  │   │   └── testwise.yaml
  │   ├── use_attributes
  │   │   ├── none.yaml (default)
  │   │   └── non_consolidated.yaml
  │   ├── use_features
  │   │   ├── none.yaml (default)
  │   │   ├── consolidated.yaml
  │   │   └── non_consolidated.yaml
  ├── algorithms
  │   ├── pre_computed_ond_agent.yaml (default)
  │   ├── pre_computed_condda_agent.yaml
  │   └── pre_computed_ond_reaction_agent.yaml

The folders in the hierarchy are considered configuration groups with the yaml
files being options that can be selected for configuration group. The defaults
or a non default configuration groups are composed to either swap abstractions,
add additional features into run, and sweep over multiple values present in the
configuration.

.. note:: Please refer to the cli section to find the configuration groups supported
          by sail-on-client

.. literalinclude:: ../../../sail_on_client/configs/algorithms/pre_computed_ond_agent.yaml
   :caption: Sample configuration for :code:`PreComputedONDAgent`
   :language: yaml

Since :code:`PreComputedONDAgent` is an smqtk algorithm, it requires additional keywords
highlighted below

.. literalinclude:: ../../../sail_on_client/configs/algorithms/pre_computed_ond_agent.yaml
   :caption: Keywords required for specifying smqtk algorithm
   :language: yaml
   :emphasize-lines: 2-4

If a default cannot be specified, :code:`???` can be used to add a requirement for
mandatory input.

.. literalinclude:: ../../../sail_on_client/configs/algorithms/pre_computed_ond_agent.yaml
   :caption: Sample specification for mandatory input
   :language: yaml
   :emphasize-lines: 5

.. Appendix 1: Links

.. _Hydra: https://hydra.cc/
.. _SMQTK: https://github.com/Kitware/SMQTK-Core
.. _configs: https://github.com/darpa-sail-on/sail-on-client/tree/master/sail_on_client/configs
.. _pre_computed_ond_agent.yaml: https://github.com/darpa-sail-on/sail-on-client/blob/master/sail_on_client/configs/algorithms/pre_computed_ond_agent.yaml
.. _pre_computed_detector: https://github.com/darpa-sail-on/sail-on-client/blob/master/sail_on_client/agent/pre_computed_detector.py
