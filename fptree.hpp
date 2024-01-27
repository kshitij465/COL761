#ifndef FPTREE_HPP
#define FPTREE_HPP

#include <cstdint>
#include <unordered_map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include "bits/stdc++.h"
#include <chrono>
#define uint64_t int

using Item = int;
using Transaction = std::vector<Item>;
using TransformedPrefixPath = std::pair<std::vector<Item>, uint64_t>;
using Pattern = std::pair<std::set<Item>, uint64_t>;


struct FPNode {
    const Item item;
    uint64_t frequency;
    std::shared_ptr<FPNode> node_link;
    std::weak_ptr<FPNode> parent;
    std::vector<std::shared_ptr<FPNode>> children;

    FPNode(const Item&, const std::shared_ptr<FPNode>&);
};

struct FPTree {
    std::shared_ptr<FPNode> root;
    std::unordered_map<Item, std::shared_ptr<FPNode>> header_table;
    uint64_t minimum_support_threshold;

    FPTree(const std::vector<Transaction>&, uint64_t);

    bool empty() const;
};


std::vector<Pattern> fptree_growth(const FPTree&,long double ran,std::chrono::system_clock::time_point &timer);


#endif  // FPTREE_HPP