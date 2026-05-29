# Use MolScope from an AI assistant (MCP)

MolScope ships an optional [Model Context Protocol](https://modelcontextprotocol.io)
(MCP) server. MCP is an open standard that lets an AI assistant call external
tools, so with this server an assistant such as Claude Code or Claude Desktop can
drive MolScope's analyses in natural language.

The server is a thin, faithful adapter over the public `molscope` API. It adds no
new science: every tool maps onto a function documented elsewhere in this user
guide. What it gives you is a conversational front end, for example:

> "Fetch trypsin (3ptb), find the benzamidine binding-site residues, and render
> a contact map."

The assistant turns that into a `binding_site` call followed by a
`render_contact_map` call and shows you the residues and the figure.

## Install

The reference MCP SDK needs Python 3.10 or newer, so the server is gated behind
an optional extra:

```bash
pip install "molscope[mcp]"
```

On Python 3.9 the extra installs nothing and `molscope-mcp` exits with a clear
hint, since the SDK is unavailable there.

## Register with a client

The server speaks MCP over stdio, which is how local clients launch it. The
console script is `molscope-mcp` (equivalently `python -m molscope.mcp_server`).

### Claude Code

```bash
claude mcp add molscope -- molscope-mcp
```

### Claude Desktop

Add an entry to the app's MCP server configuration:

```json
{
  "mcpServers": {
    "molscope": {
      "command": "molscope-mcp"
    }
  }
}
```

Point `command` at the `molscope-mcp` executable from the environment where you
installed `molscope[mcp]` (use its absolute path if the client does not share
your shell's `PATH`).

## Tools

Every tool takes a `source` that is either a path to a local coordinate file
(`.pdb`, `.cif`, `.xyz`, `.sdf`, optionally gzipped) or a 4-character RCSB PDB id
that is fetched and cached.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `summarize_structure` | `source` | One-line summary: atoms, formula, chains, size. |
| `compute_descriptors` | `sources` (list), `preset` | JSON descriptor table, one row per structure. The batch tool. |
| `secondary_structure` | `source` | JSON per-residue DSSP codes and helix/strand/coil composition. |
| `contact_map` | `source`, `cutoff`, `level`, `method`, `min_seq_sep` | JSON contact count, contact order, and labelled contacting pairs. |
| `binding_site` | `source`, `ligand`, `cutoff` | JSON binding-site residues ordered closest-first. |
| `molecular_graph` | `source`, `preset`, `include_chemical_features` | JSON node/edge counts and feature names. |
| `coarse_grain` | `source`, `mapping` | JSON bead-assignment statistics. |
| `render_structure` | `source`, `color_by` | PNG of the 3D view. |
| `render_contact_map` | `source`, `cutoff`, `level`, `method` | PNG heatmap. |

Data tools return JSON text so the model can read the values directly; the two
render tools return PNG images. Large per-residue or per-pair lists are truncated
with a `*_truncated` flag so a big structure cannot flood the conversation.

## Scope

The server intentionally wraps only existing read-only analyses. It does not
write files, mutate structures, or add capabilities beyond the library, and it
inherits every limitation documented in
[Limitations by workflow](../limitations.md). For scripted or batch use, prefer
the Python API or the `molscope` command-line interface; the MCP server is for
interactive, assistant-driven exploration.
