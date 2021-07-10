# allot_routing_table

[Allot Routing Table](https://www.hariguchi.org/art/art.pdf) in Rust (and maybe Go too).


#### What is an Allot Routing Table?

"Allot" is a strange name the paper uses for a function to modify some elements in an array, and
can effectively be ignored, the exact wording isn't well explained in the paper and I have no
idea what is meant by it.

A routing table, on the other hand is much more well-defined. It is essentially a set that
allows for efficient prefix matching for IP addresses (but also can work for more general
purpose things).

Let's go through an example, as the paper does, but a little more clearly.
Consider all 4 bit possibilities: `0b0000`, `0b0001`, etc.

The operations possible with a routing table are:

- `Search(table, 4 bits)`:

Given a table with some entries, we would like to find the most specific entry that matches the
given set of bits. Consider a table with both `0b1000`, and `0b1111`. If we perform a search for
`0b1110`, we match the first 3 bit from both entries, but since the last bit does not match
`0b1111`, the most specific entry we have is `0b1000`.

In summary, search returns the longest matching result in the table currently.

- `Insert(table, 4 bits, # prefix bits) -> did add new entry to table`

This adds a route to the table, with some number of prefix bits to match on.
To make this more clear, consider adding: `0b1000`, with 1 prefix bit.
This means, when searching, match on any item `0b1***`, and return `0b1000`.

We would like to be able to match the most specific route though, so we can later add more
routes such as `0b1111` with 4 prefix bits. If we perform a search for `0b1111`, we should
expect to see that the table has `0b1111`, even though it matches both `0b1000`, and `0b1111`,
because `0b1111` is more specific.

- `Remove(table, 4 bits, # prefix) -> Old entry or null`:

This is the opposite of insert, and removes an entry from the table, looking for exact matches
only.

---

Of course, only having it work for 4 bits is not so useful.
This implements allot routing tables for IPv4 and IPv6, and can be extended for arbitrary
metadata on the items.

## TODO

Add some benchmarks. I hope this implementation is blazing fast, but have to see.
