#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]
use std::fmt::Debug;
use std::net::{Ipv4Addr, Ipv6Addr};

// A Route represents a series of bytes which has a fixed width.
pub trait Route: Eq + Copy + Debug {
    const BITS: usize;
    fn addr(&self) -> [u8; (Self::BITS + 7) / 8];
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RouteIpv4(pub Ipv4Addr);
impl Route for RouteIpv4 {
    const BITS: usize = 32;
    #[inline]
    fn addr(&self) -> [u8; 4] {
        self.0.octets()
    }
}
impl From<Ipv4Addr> for RouteIpv4 {
    fn from(ip: Ipv4Addr) -> Self {
        Self(ip)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RouteIpv6(pub Ipv6Addr);
impl Route for RouteIpv6 {
    const BITS: usize = 128;
    #[inline]
    fn addr(&self) -> [u8; 16] {
        self.0.octets()
    }
}
impl RouteIpv6 {
    pub fn new(ip: Ipv6Addr) -> Self {
        Self(ip)
    }
}

impl From<Ipv6Addr> for RouteIpv6 {
    fn from(v: Ipv6Addr) -> Self {
        Self(v)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Route4Bit(pub u8);
impl Route for Route4Bit {
    const BITS: usize = 4;
    #[inline]
    fn addr(&self) -> [u8; 1] {
        [self.0]
    }
}

impl From<u8> for Route4Bit {
    fn from(v: u8) -> Self {
        assert!(v < (1 << 4));
        Self(v)
    }
}

fn allot<R: Eq + Copy>(x: &mut [R], smallest_fringe_idx: usize, base_index: usize, old: R, new: R) {
    let b = base_index;
    let t = smallest_fringe_idx;
    if x[b] == old {
        x[b] = new;
    } else {
        return;
    }
    if b >= t {
        return;
    }
    let b = b << 1;
    allot(x, t, b, old, new);
    allot(x, t, b + 1, old, new);
}
/*
*/

/*
// Non-recursive implementation is borken
fn allot<R: Eq + Copy>(
    x: &mut [R],
    t: usize,
    b: usize,
    old: R,
    new: R,
) {
    if b >= t {
        if x[b] == old {
            x[b] = new;
        }
        return;
    }

    start_change(x, b, t, b, old, new)
}

fn start_change<R: Eq + Copy>(x: &mut [R], mut j: usize, t: usize, b: usize, old: R, new: R) {
    j <<= 1;
    if j < t {
        return non_fringe(x, j, t, b, old, new);
    }
    loop {
      if x[b] == old {
        x[b] = new;
      }
      if j & 1 == 1 {
        return move_up(x, j, t, b, old, new);
      }
      j += 1;
    }
}

fn non_fringe<R: Eq + Copy>(x: &mut [R], j: usize, t: usize, b: usize, old: R, new: R) {
    if x[j] == old {
        start_change(x, j, t, b, old, new);
    } else {
        move_on(x, j, t, b, old, new);
    }
}

fn move_on<R: Eq + Copy>(x: &mut [R], j: usize, t: usize, b: usize, old: R, new: R) {
    if j & 1 == 1 {
        move_up(x, j, t, b, old, new);
    } else {
        non_fringe(x, j + 1, t, b, old, new)
    }
}
fn move_up<R: Eq + Copy>(x: &mut [R], j: usize, t: usize, b: usize, old: R, new: R) {
    let j = j >> 1;
    x[j] = new;
    if j != b {
        move_on(x, j, t, b, old, new);
    }
}
*/

const fn base_index(w: usize, a: usize, l: usize) -> usize {
    (a >> (w - l)) | (1 << l)
}

const fn fringe_index(width: usize, addr: usize) -> usize {
    base_index(width, addr, width)
}

fn insert_s<R: Route>(x: &mut [Option<R>], width: usize, addr: usize, prefix: usize, r: R) -> bool {
    let b = base_index(width, addr, prefix);
    let curr = x[b];
    if curr == Some(r) {
        return false;
    }
    allot(x, 1 << width, b, curr, Some(r));
    true
}

fn remove_s<R: Route>(x: &mut [Option<R>], width: usize, addr: usize, prefix: usize) -> Option<R> {
    let b = base_index(width, addr, prefix);
    let old = x[b as usize]?;
    allot(x, 1 << width, b, Some(old), x[b as usize >> 1]);
    Some(old)
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SingleLevelTable<R> {
    routes: Vec<Option<R>>,
    len: usize,
}

// assumes that offset is divisible by 8.
#[inline]
fn get_bits(byte_offset: usize, bits: usize, src: &[u8]) -> u32 {
    if byte_offset > src.len() {
        return 0;
    }

    let mask = (1 << bits) - 1;
    let v = match &src[..src.len() - byte_offset] {
        &[] => return 0,
        &[d] => d as u32,
        &[c, d] => (((c as u32) << 8) + (d as u32)),
        &[b, c, d] => (((b as u32) << 16) + ((c as u32) << 8) + d as u32),
        &[.., a, b, c, d] => {
            ((a as u32) << 24) + ((b as u32) << 16) + ((c as u32) << 8) + (d as u32)
        }
    };
    v & mask
}

impl<R: Route> SingleLevelTable<R>
where
    [(); (R::BITS + 7) / 8]: ,
{
    fn new(width: usize) -> Self {
        assert!(width <= 32 && width > 0);
        let entries = 1 << (width + 1);
        Self {
            routes: vec![None; entries],
            len: 0,
        }
    }
    pub fn insert(&mut self, r: impl Into<R>, prefix_len: usize) -> bool {
        let r = r.into();
        assert!(prefix_len <= R::BITS);
        insert_s(
            &mut self.routes,
            R::BITS,
            get_bits(0, R::BITS, &r.addr()) as usize,
            prefix_len,
            r,
        )
    }
    pub fn search(&self, r: impl Into<R>) -> Option<&R> {
        let r = r.into();
        (&self.routes[fringe_index(R::BITS, get_bits(0, R::BITS, &r.addr()) as usize)]).as_ref()
    }
    pub fn remove(&mut self, r: impl Into<R>, prefix_len: usize) -> Option<R> {
        let r = r.into();
        assert!(prefix_len <= R::BITS);
        remove_s(
            &mut self.routes,
            R::BITS,
            get_bits(0, R::BITS, &r.addr()) as usize,
            prefix_len,
        )
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
enum TableInner<R> {
    Leaf(SingleLevelTable<R>),
    Multi(MultiLevelTable<R>),
}

impl<R> TableInner<R> {
    fn mut_routes(&mut self) -> &mut [Option<R>] {
        match self {
            TableInner::Leaf(s) => &mut s.routes[..],
            TableInner::Multi(m) => &mut m.routes[..],
        }
    }
    fn routes(&self) -> &[Option<R>] {
        match self {
            TableInner::Leaf(s) => &s.routes[..],
            TableInner::Multi(m) => &m.routes[..],
        }
    }
    fn mut_children(&mut self) -> Option<&mut [Option<TableInner<R>>]> {
        if let TableInner::Multi(m) = self {
            Some(&mut m.children[..])
        } else {
            None
        }
    }
    fn mut_children_and_len(&mut self) -> (Option<&mut [Option<TableInner<R>>]>, &mut usize) {
        match self {
            TableInner::Leaf(s) => (None, &mut s.len),
            TableInner::Multi(m) => (Some(&mut m.children[..]), &mut m.len),
        }
    }
    fn children(&self) -> Option<&[Option<TableInner<R>>]> {
        if let TableInner::Multi(m) = self {
            Some(&m.children[..])
        } else {
            None
        }
    }
    pub fn len(&self) -> usize {
        match self {
            TableInner::Leaf(s) => s.len,
            TableInner::Multi(m) => m.len,
        }
    }
    fn len_mut(&mut self) -> &mut usize {
        match self {
            TableInner::Leaf(s) => &mut s.len,
            TableInner::Multi(m) => &mut m.len,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct MultiLevelTable<R> {
    routes: Vec<Option<R>>,
    children: Vec<Option<TableInner<R>>>,
    len: usize,
}

impl<R: Route> MultiLevelTable<R> {
    fn new(width: usize) -> Self {
        let entries = 1 << (width + 1);
        Self {
            routes: vec![None; entries],
            children: vec![None; entries],
            len: 0,
        }
    }
}

#[derive(Clone)]
pub struct Table<R> {
    strides: Vec<usize>,
    root: TableInner<R>,

    idxs_from_root: Vec<usize>,

    len: usize,
}

impl<R: Route> Table<R>
where
    [(); (R::BITS + 7) / 8]: ,
{
    pub fn new(strides: Vec<usize>) -> Self {
        assert!(strides.len() > 0, "Must pass at least 1 stride");
        assert_eq!(
            strides.iter().sum::<usize>(),
            R::BITS,
            "strides do not sum to total width"
        );
        assert!(
            strides.iter().all(|v| v % 8 == 0),
            "All strides must be byte multiples"
        );

        let root = if strides.len() == 1 {
            TableInner::Leaf(SingleLevelTable::new(strides[0]))
        } else {
            TableInner::Multi(MultiLevelTable::new(strides[0]))
        };
        let levels = strides.len();
        Self {
            strides,
            root,
            idxs_from_root: vec![0; levels - 1],
            len: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn insert(&mut self, r: impl Into<R>, prefix_len: usize) -> bool {
        let r = r.into();
        assert!(prefix_len <= R::BITS);

        let mut curr = &mut self.root;

        // degenerate case of empty route
        if r.addr().iter().all(|&v| v == 0) && prefix_len == 0 {
            let routes = curr.mut_routes();
            if routes[1].is_some() {
                return false;
            }
            routes[1] = Some(r);
            return true;
        }

        let mut ss: usize = 0;
        let mut level = 0;
        let (stride, curr_stride) = loop {
            let curr_stride = self.strides[level];
            ss += curr_stride;
            let left_shift = R::BITS - ss;
            assert!(left_shift % 8 == 0);
            let stride = get_bits(left_shift / 8, curr_stride, &r.addr()) as usize;
            if prefix_len <= ss {
                break (stride, curr_stride);
            }
            let i = fringe_index(curr_stride, stride);

            let (children, len) = curr.mut_children_and_len();
            let children = children.unwrap();
            assert!(level + 1 < self.strides.len());
            let next_stride = self.strides[level + 1];
            let child = &mut children[i];
            if let Some(child) = child {
              curr = child
            } else {
                *len += 1;
                curr = child.insert(TableInner::Multi(MultiLevelTable::new(next_stride)));
            };
            level += 1;
        };
        ss -= curr_stride;
        let did_insert = insert_s(curr.mut_routes(), curr_stride, stride, prefix_len - ss, r);

        if did_insert {
            *curr.len_mut() += 1;

            self.len += 1
        }
        did_insert
    }
    pub fn remove(&mut self, r: impl Into<R>, prefix_len: usize) -> Option<R> {
        let r = r.into();
        assert!(prefix_len <= R::BITS);

        let mut curr = &mut self.root;
        if r.addr().iter().all(|&v| v == 0) && prefix_len == 0 {
            return curr.mut_routes().iter_mut().nth(1)?.take();
        }

        let mut ss: usize = 0;
        let mut level = 0;
        let (stride, curr_stride) = loop {
            assert!(level < self.strides.len());

            let curr_stride = self.strides[level];
            ss += curr_stride;
            let left_shift = R::BITS - ss;
            // assert!(left_shift % 8 == 0);
            let stride = get_bits(left_shift / 8, curr_stride, &r.addr()) as usize;
            if prefix_len as usize <= ss {
                break (stride, curr_stride);
            }
            let i = fringe_index(curr_stride, stride);
            self.idxs_from_root[level] = i;
            // parent_idxs.push(i);
            let children = curr.mut_children()?;
            curr = children[i].as_mut()?;
            level += 1;
        };

        ss -= curr_stride;
        let old = remove_s(curr.mut_routes(), curr_stride, stride, prefix_len - ss)?;

        *curr.len_mut() -= 1;
        while level > 0 && curr.len() == 0 {
            level -= 1;
            let to_rm = self.idxs_from_root[level];
            curr = self.mut_child_at_level(level);
            curr.mut_children().unwrap()[to_rm] = None;
            *curr.len_mut() -= 1;
        }
        self.len -= 1;
        Some(old)
    }

    fn mut_child_at_level(&mut self, level: usize) -> &mut TableInner<R> {
        assert!(level < self.strides.len());
        let mut curr = &mut self.root;
        for _ in 0..level {
            let children = curr.mut_children().unwrap();
            curr = children[self.idxs_from_root[level]].as_mut().unwrap();
        }
        curr
    }

    pub fn search(&self, r: impl Into<R>) -> Option<R> {
        let r = r.into();

        let mut curr = &self.root;
        // lmr = longest matching result :)
        let mut lmr = curr.routes()[1];

        let mut level = 0;
        let mut ss = 0;
        loop {
            let curr_stride = self.strides[level];
            ss += curr_stride;
            let left_shift = R::BITS - ss;
            assert!(left_shift % 8 == 0);
            let stride = get_bits(left_shift / 8, curr_stride, &r.addr()) as usize;
            let i = fringe_index(curr_stride, stride);
            let children = if let Some(c) = curr.children() {
                c
            } else {
                return lmr;
            };
            let curr_lmr = curr.routes()[i];
            lmr = curr_lmr.or(lmr);
            if let Some(child) = &children[i] {
                curr = child;
                level += 1
            } else {
                return lmr;
            }
        }
    }
}

impl Table<RouteIpv4> {
    pub fn new_ipv4(strides: Vec<usize>) -> Self {
        Self::new(strides)
    }
}

impl Table<RouteIpv6> {
    pub fn new_ipv6(strides: Vec<usize>) -> Self {
        Self::new(strides)
    }
    pub fn new_ipv6_default() -> Self {
        Self::new(vec![8; 16])
    }
}

#[test]
fn test_single_level_simple() {
    let mut t = SingleLevelTable::<Route4Bit>::new(4);
    assert!(t.insert(12, 2));
    assert!(t.search(12).is_some());
    assert_eq!(t.remove(12, 2), Some(12.into()));
    assert_eq!(t.search(12), None);

    assert!(t.insert(0b1000, 1));
    assert!(t.search(0b1001).is_some());
    assert!(t.search(0b0001).is_none());
    assert!(t.search(0b1111).is_some());
    assert!(t.insert(0b1111, 4));
    assert_eq!(t.search(0b1111), Some(&0b1111.into()));
    assert_eq!(t.search(0b1110), Some(&0b1000.into()));
    assert_eq!(t.search(0b1000), Some(&0b1000.into()));
}

#[test]
fn test_multi_level_ipv4() {
    let mut t = Table::new_ipv4(vec![8; 4]);
    assert!(t.insert(Ipv4Addr::new(127, 0, 0, 1), 12));

    assert!(t.search(Ipv4Addr::new(127, 0, 0, 1)).is_some());

    assert!(t.remove(Ipv4Addr::new(127, 0, 0, 1), 8).is_none());
    assert!(t.remove(Ipv4Addr::new(127, 0, 0, 1), 16).is_some());
}

#[test]
fn test_multi_level_ipv6() {
    let mut t = Table::new_ipv6(vec![8; 16]);
    assert!(t.insert(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 12));

    assert!(t.search(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0)).is_some());
    assert!(t.search(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0)).is_some());

    assert_eq!(t.remove(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 8), None);
    assert!(t
        .remove(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 16)
        .is_some());
}

#[test]
fn test_insert_ipv4_exhaustive() {
    let mut t = Table::new_ipv4(vec![8, 8, 8, 8]);
    for a in 0..=255u8 {
        // it's too expensive to test all ip addresses, but can test a lot
        for b in 0..=100u8 {
            let ip = Ipv4Addr::new(a, b, 0, 0);
            assert!(t.insert(ip, 32))
        }
    }
}
