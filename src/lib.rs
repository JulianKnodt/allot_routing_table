#![feature(const_generics, const_evaluatable_checked)]
#![allow(unused, incomplete_features)]
use std::mem;
use std::net::{Ipv4Addr, Ipv6Addr};

pub trait Route: Eq + Copy + std::fmt::Debug {
    const BITS: usize;
    fn addr(&self) -> [u8; (Self::BITS + 7) / 8];
    fn prefix_len(&self) -> u8;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RouteIpv4(Ipv4Addr, u8);
impl Route for RouteIpv4 {
    const BITS: usize = 32;
    #[inline]
    fn addr(&self) -> [u8; 4] {
        self.0.octets()
    }
    #[inline]
    fn prefix_len(&self) -> u8 {
        self.1
    }
}
impl RouteIpv4 {
    pub fn new(ip: Ipv4Addr, prefix_len: u8) -> Self {
        assert!(prefix_len <= 32);
        Self(ip, prefix_len)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RouteIpv6(Ipv6Addr, u8);
impl Route for RouteIpv6 {
    const BITS: usize = 128;
    #[inline]
    fn addr(&self) -> [u8; 16] {
        self.0.octets()
    }
    #[inline]
    fn prefix_len(&self) -> u8 {
        self.1
    }
}
impl RouteIpv6 {
    pub fn new(ip: Ipv6Addr, prefix_len: u8) -> Self {
        assert!(prefix_len <= 128);
        Self(ip, prefix_len)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Route4Bit(u8, u8);
impl Route for Route4Bit {
    const BITS: usize = 4;
    #[inline]
    fn addr(&self) -> [u8; 1] {
        [self.0]
    }
    #[inline]
    fn prefix_len(&self) -> u8 {
        self.1
    }
}

impl Route4Bit {
    pub fn new(addr: u8, prefix_len: u8) -> Self {
        assert!(addr < (1 << 4));
        assert!(prefix_len <= 4);
        Self(addr, prefix_len)
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

const fn base_index(w: usize, a: usize, l: usize) -> usize {
    (a >> (w - l)) | (1 << l)
}

const fn fringe_index(width: usize, addr: usize) -> usize {
    base_index(width, addr, width)
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct SingleLevelTable<R> {
    routes: Vec<Option<R>>,
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
    let next = x[b as usize >> 1];
    allot(x, 1 << width, b, Some(old), next);
    Some(old)
}

// assumes that offset is divisible by 8.
#[inline]
fn get_bits(byte_offset: usize, bits: usize, mut src: &[u8]) -> u32 {
    if byte_offset > src.len() {
        return 0;
    }

    let mask = ((1 << bits) - 1);
    let v = match &src[..src.len() - byte_offset] {
        &[] => return 0,
        &[d] => d as u32,
        &[c, d] => (((c as u32) << 8) + (d as u32)),
        &[b, c, d] => (((b as u32) << 16) + ((c as u32) << 8) + d as u32),
        &[.., a, b, c, d] => {
            (((a as u32) << 24) + ((b as u32) << 16) + ((c as u32) << 8) + (d as u32))
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
        }
    }
    fn insert(&mut self, bits: usize, r: R) -> bool {
        insert_s(
            &mut self.routes,
            bits,
            get_bits(0, bits, &r.addr()) as usize,
            r.prefix_len() as usize,
            r,
        )
    }
    fn search(&self, bits: usize, addr: &[u8]) -> Option<&R> {
        (&self.routes[fringe_index(bits, get_bits(0, bits, addr) as usize)]).as_ref()
    }
    fn remove(&mut self, bits: usize, r: R) -> Option<R> {
        remove_s(
            &mut self.routes,
            bits,
            get_bits(0, bits, &r.addr()) as usize,
            r.prefix_len() as usize,
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
    fn children(&self) -> Option<&[Option<TableInner<R>>]> {
        if let TableInner::Multi(m) = self {
            Some(&m.children[..])
        } else {
            None
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct MultiLevelTable<R> {
    routes: Vec<Option<R>>,
    children: Vec<Option<TableInner<R>>>,
}

impl<R: Route> MultiLevelTable<R> {
    fn new(width: usize) -> Self {
        let entries = 1 << (width + 1);
        Self {
            routes: vec![None; entries],
            children: vec![None; entries],
        }
    }
}

#[derive(Clone)]
pub struct Table<R> {
    strides: Vec<usize>,
    root: TableInner<R>,

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
        Self { strides, root, len: 0 }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn insert(&mut self, r: R) -> bool {
        let mut level = 0;
        let mut ss: usize = 0;
        let mut stride;

        let mut curr = &mut self.root;
        let routes = curr.mut_routes();

        // degenerate case of empty route
        if r.addr().iter().all(|&v| v == 0) && r.prefix_len() == 0 {
            if routes[1].is_some() {
                return false;
            }
            routes[1] = Some(r);
            return true;
        }
        loop {
            let curr_stride = self.strides[level];
            ss += curr_stride;
            let left_shift = R::BITS - ss;
            assert!(left_shift % 8 == 0);
            stride = get_bits(left_shift / 8, curr_stride, &r.addr()) as usize;
            if r.prefix_len() as usize <= ss {
                break;
            }
            let i = fringe_index(curr_stride, stride);

            let children = curr.mut_children().unwrap();
            let next_stride = self.strides[level + 1];
            let child = children[i]
                .get_or_insert_with(|| TableInner::Multi(MultiLevelTable::new(next_stride)));
            curr = child;
            level += 1;
        }
        let curr_stride = self.strides[level];
        ss -= curr_stride;
        let routes = match curr {
            TableInner::Leaf(s) => &mut s.routes,
            TableInner::Multi(m) => &mut m.routes,
        };
        let did_insert = insert_s(routes, curr_stride, stride, r.prefix_len() as usize - ss, r);

        if did_insert {
            //curr.refs += 1;

            self.len += 1
        }
        did_insert
    }
    pub fn remove(&mut self, r: R) -> Option<R> {
        let mut level = 0;
        let mut parent_idxs = vec![];
        let mut ss = 0;
        let mut stride;

        let mut curr = &mut self.root;
        let routes = curr.mut_routes();
        if r.addr().iter().all(|&v| v == 0) && r.prefix_len() == 0 {
            return routes[1].take();
        }

        loop {
            let curr_stride = self.strides[level];
            ss += curr_stride;
            let left_shift = R::BITS - ss;
            assert!(left_shift % 8 == 0);
            stride = get_bits(left_shift / 8, curr_stride, &r.addr()) as usize;
            if r.prefix_len() as usize <= ss {
                break;
            }
            let i = fringe_index(curr_stride, stride);
            parent_idxs.push(i);
            let children = curr.mut_children()?;
            curr = children[i].as_mut()?;
            level += 1;
        }

        let curr_stride = self.strides[level];
        ss -= curr_stride;
        let routes = match curr {
            TableInner::Leaf(s) => &mut s.routes,
            TableInner::Multi(m) => &mut m.routes,
        };
        let old = remove_s(routes, curr_stride, stride, r.prefix_len() as usize - ss)?;
        // TODO free stuff here
        self.len -= 1;
        Some(old)
    }

    pub fn search(&self, r: R) -> Option<R> {
        let mut level = 0;
        let mut ss = 0;
        let mut stride;

        let mut curr = &self.root;
        // lmr = longest matching result :)
        let mut lmr = curr.routes()[1];

        loop {
            let curr_stride = self.strides[level];
            ss += curr_stride;
            let left_shift = R::BITS - ss;
            assert!(left_shift % 8 == 0);
            stride = get_bits(left_shift / 8, curr_stride, &r.addr()) as usize;
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
    fn new_ipv4(strides: Vec<usize>) -> Self {
        Self::new(strides)
    }
}

impl Table<RouteIpv6> {
    fn new_ipv6(strides: Vec<usize>) -> Self {
        Self::new(strides)
    }
    fn new_ipv6_default() -> Self {
        Self::new(vec![8; 16])
    }
}

#[test]
fn test_single_level_simple() {
    let mut t = SingleLevelTable::<Route4Bit>::new(4);
    assert!(t.insert(4, Route4Bit::new(12, 2)));
    assert!(t.search(4, &[12]).is_some());
    assert!(t.delete(4, Route4Bit::new(12, 2)).is_some());
    assert!(t.search(4, &[12]).is_none());

    assert!(t.insert(4, Route4Bit::new(0b1000, 1)));
    assert!(t.search(4, &[0b1001]).is_some());
    assert!(t.search(4, &[0b0001]).is_none());
    assert!(t.search(4, &[0b1111]).is_some());
    assert!(t.insert(4, Route4Bit::new(0b1111, 4)));
    assert_eq!(t.search(4, &[0b1111]), Some(&Route4Bit::new(0b1111, 4)));
    assert_eq!(t.search(4, &[0b1110]), Some(&Route4Bit::new(0b1000, 1)));
    assert_eq!(t.search(4, &[0b1000]), Some(&Route4Bit::new(0b1000, 1)));
}

#[test]
fn test_multi_level_ipv4() {
    let mut t = Table::new_ipv4(vec![8; 4]);
    assert!(t.insert(RouteIpv4::new(Ipv4Addr::new(127, 0, 0, 1), 12)));

    assert_ne!(
        t.search(RouteIpv4::new(Ipv4Addr::new(127, 0, 0, 1), 12)),
        None,
    );
    assert_ne!(
        t.search(RouteIpv4::new(Ipv4Addr::new(127, 0, 0, 1), 16)),
        None
    );
    assert_ne!(
        t.search(RouteIpv4::new(Ipv4Addr::new(127, 0, 0, 1), 8)),
        None
    );

    assert_eq!(
        t.remove(RouteIpv4::new(Ipv4Addr::new(127, 0, 0, 1), 8)),
        None
    );
    assert_ne!(
        t.remove(RouteIpv4::new(Ipv4Addr::new(127, 0, 0, 1), 16)),
        None
    );
}

#[test]
fn test_multi_level_ipv6() {
    let mut t = Table::new_ipv6(vec![8; 16]);
    assert!(t.insert(RouteIpv6::new(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 12)));

    assert_ne!(
        t.search(RouteIpv6::new(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 12)),
        None,
    );
    assert_ne!(
        t.search(RouteIpv6::new(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 16)),
        None
    );
    assert_ne!(
        t.search(RouteIpv6::new(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 8)),
        None
    );

    assert_eq!(
        t.remove(RouteIpv6::new(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 8)),
        None
    );
    assert_ne!(
        t.remove(RouteIpv6::new(Ipv6Addr::new(127, 0, 0, 1, 0, 0, 0, 0), 16)),
        None
    );
}

#[test]
fn test_insert_ipv4_exhaustive() {
    let mut t = Table::new_ipv4(vec![8, 8, 8, 8]);
    for a in 0..=255u8 {
        // it's too expensive to test all ip addresses, but can test a lot
        for b in 0..=100u8 {
            let ip = Ipv4Addr::new(a, b, 0, 0);
            assert!(t.insert(RouteIpv4::new(ip, 32)))
        }
    }
}
