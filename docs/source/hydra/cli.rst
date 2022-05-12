Command Line Interface
======================

The command line interface for sail-on-client is built upon `Hydra`_. This allows
the CLI to be a source for providing configuration parameters required in a experiment.

Usage
-----

List of configuration group and default configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..code-block::

    sail-on-client --help

.. code-block::
   :caption: Output for help

    == Configuration groups ==
    Compose your configuration from those groups (group=option)

    algorithms: override_agent, pre_computed_condda_agent, pre_computed_ond_agent, pre_computed_ond_reaction_agent
    base: base_condda, base_ond
    harness: local, override_harness, par
    protocol: condda, ond, visual
    protocol/detection: given, system
    protocol/eval: none, with_reaction, without_reaction
    protocol/feedback: classification, detection, detection_and_classification, none, score
    protocol/resume_session: none, resume
    protocol/save_attributes: none, save_attributes
    protocol/save_features: elementwise, none, testwise
    protocol/use_attributes: non_consolidated, none
    protocol/use_features: consolidated, non_consolidated, none

    == Config ==
    Override anything in the config (foo.bar=value)

    protocol:
      smqtk:
        config:
          dataset_root: ???
          domain: ???
          save_dir: ???
          seed: 4001
          test_ids:
          - ???
          is_eval_enabled: false
          is_eval_roundwise_enabled: false
          has_baseline: false
          has_reaction_baseline: false
          baseline_class: ''
          use_feedback: false
          feedback_type: null
          hints: null
          resume_session: false
          resume_session_ids: null
          save_elementwise: false
          save_features: false
          feature_extraction_only: false
          feature_dir: ''
          use_saved_features: false
          use_consolidated_features: false
          use_saved_attributes: false
          algorithms:
            PreComputedONDAgent:
              smqtk:
                class: PreComputedONDAgent
                config:
                  algorithm_name: ???
                  cache_dir: ???
                  has_roundwise_file: false
                  round_size: ???
          harness:
            smqtk:
              class: LocalHarness
              config:
                data_dir: ???
                result_dir: ???
                gt_dir: ???
                gt_config: ???
        class: ONDProtocol

List of Defaults
^^^^^^^^^^^^^^^^

.. code-block::

   sail-on-client --info defaults

.. code-block::
   :caption: Output for defaults

    Defaults List
    *************
    | Config path                       | Package                          | _self_ | Parent        |
    --------------------------------------------------------------------------------------------------
    | hydra/output/default              | hydra                            | False  | hydra/config  |
    | hydra/launcher/basic              | hydra.launcher                   | False  | hydra/config  |
    | hydra/sweeper/basic               | hydra.sweeper                    | False  | hydra/config  |
    | hydra/help/default                | hydra.help                       | False  | hydra/config  |
    | hydra/hydra_help/default          | hydra.hydra_help                 | False  | hydra/config  |
    | hydra/hydra_logging/colorlog      | hydra.hydra_logging              | False  | hydra/config  |
    | hydra/job_logging/colorlog        | hydra.job_logging                | False  | hydra/config  |
    | hydra/env/default                 | hydra.env                        | False  | hydra/config  |
    | hydra/config                      | hydra                            | True   | <root>        |
    | protocol/visual                   | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/eval/none                | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/feedback/none            | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/detection/system         | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/resume_session/none      | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/save_features/none       | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/use_features/none        | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/use_attributes/none      | protocol.smqtk.config            | False  | protocol/ond  |
    | protocol/ond                      | protocol.smqtk.config            | True   | base/base_ond |
    | algorithms/pre_computed_ond_agent | protocol.smqtk.config.algorithms | False  | base/base_ond |
    | harness/local                     | protocol.smqtk.config.harness    | False  | base/base_ond |
    | base/base_ond                     | protocol.smqtk                   | True   | default       |
    | default                           |                                  | True   | <root>        |
    --------------------------------------------------------------------------------------------------


Running Experiments
^^^^^^^^^^^^^^^^^^^

..code-block::

    sail-on-client --config-dir configs/ \
                   --config-name <name of the config under config directory without .yaml> \
                   Mandatory <key>=<value> pairs seperated by space \
                   Overriding <key>=<value> pairs seperated by space


.. Appendix 1: Links

.. _Hydra: https://hydra.cc/
