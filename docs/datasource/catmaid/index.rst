.. _catmaid-datasource:

CATMAID
=======

The CATMAID data service driver exposes skeletons from a CATMAID project,
including spatially indexed skeleton loading and optional editing.

URL syntax
----------

- :file:`catmaid://https://{host}/{project-id}`
- :file:`catmaid://http://{host}/{project-id}`

Browser authentication
----------------------

Neuroglancer first requests CATMAID's anonymous API token so public projects
continue to load without interaction. If the anonymous account cannot access a
project, Neuroglancer asks for a personal CATMAID API token.

Open the CATMAID server, use the account menu to obtain an API token, and paste
it into the Neuroglancer prompt. The token is cached for the current browser tab
under the CATMAID base URL, allowing other projects on the same server to reuse
it. A rejected token is removed automatically and requested again. Tokens are
not included in datasource URLs or serialized viewer state.

Because an API token has the same permissions as its CATMAID account, use HTTPS
for authenticated deployments and do not share the token.

Python authentication
---------------------

Python-hosted viewers obtain credentials on the Python server. Configure a
token before adding CATMAID layers:

.. code-block:: python

   import neuroglancer

   neuroglancer.set_catmaid_token(
       "https://catmaid.example",
       "your-api-token",
   )

Pass ``None`` as the token to remove a configured value. Deployments may
instead set ``CATMAID_CREDENTIALS`` to a JSON object keyed by CATMAID base
URL:

.. code-block:: shell

   export CATMAID_CREDENTIALS='{"https://catmaid.example": "your-api-token"}'

Tokens configured with ``set_catmaid_token`` take precedence over the
environment. If neither is present, the Python provider attempts anonymous
access.

Segment properties
------------------

Each skeleton carries two segment properties: the CATMAID neuron name as the
segment label, and the neuron's annotations as a ``string`` property named
``annotations``, sorted and joined by single spaces. They are read once per
project with ``skeleton/neuronnames`` and ``annotations/forskeletons``, and
read again for the affected skeletons after a merge or a split.

The segment search matches free text as a prefix of the label or of the
annotations, and a leading ``/`` as a regular expression over both. A query
made only of digits is a segment id lookup, so search for an annotation such
as ``0000001`` with ``/0000001``. The annotations of the selected segment are
listed in the selection details panel.

Server requirements
-------------------

The CATMAID deployment must allow the Neuroglancer origin to make cross-origin
``GET`` and ``POST`` requests and must allow the ``Authorization`` and
``Content-Type`` request headers. Project read and edit access is determined
by the CATMAID account associated with the token. Separately, Neuroglancer only
enables editing when the linked stack metadata sets ``read_only`` to
``false``; see :ref:`skeleton-editing-sources`.
