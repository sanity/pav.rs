# Pair Adjacent Violators

## Overview

An implementation of the [Pair Adjacent Violators](http://gifi.stat.ucla.edu/janspubs/2009/reports/deleeuw_hornik_mair_R_09.pdf) algorithm for [isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression). Note this algorithm is also known as "Pool Adjacent Violators".

### What is "Isotonic Regression" and why should I care?

Imagine you have two variables, _x_ and _y_, and you don't know the relationship between them, but you know that if _x_ increases then _y_ will increase, and if _x_ decreases then _y_ will decrease.  Alternatively it may be the opposite, if _x_ increases then _y_ decreases, and if _x_ decreases then _y_ increases.

Examples of such isotonic or monotonic relationships include:

 * _x_ is the pressure applied to the accelerator in a car, _y_ is the acceleration of the car (acceleration increases as more pressure is applied)
 * _x_ is the rate at which a web server is receiving HTTP requests, _y_ is the CPU usage of the web server (server CPU usage will increase as the request rate increases)
 * _x_ is the price of an item, and _y_ is the probability that someone will buy it (this would be a decreasing relationship, as _x_ increases _y_ decreases)

These are all examples of an isotonic relationship between two variables, where the relationship is likely to be more complex than linear.

So we know the relationship between _x_ and _y_ is isotonic, and let's also say that we've been able to collect data about actual _x_ and _y_ values that occur in practice.

What we'd really like to be able to do is estimate, for any given _x_, what _y_ will be, or alternatively for any given _y_, what _x_ would be required.

But of course real-world data is noisy, and is unlikely to be strictly isotonic, so we want something that allows us to feed in this raw noisy data, figure out the actual relationship between _x_ and _y_, and then use this to allow us to predict _y_ given _x_, or to predict what value of _x_ will give us a particular value of _y_.  This is the purpose of the pair-adjacent-violators algorithm.

#### ...and why should I care?

Using the examples I provide above:

* A self-driving car could use it to learn how much pressure to apply to the accelerator to give a desired amount of acceleration
* An autoscaling system could use it to help predict how many web servers they need to handle a given amount of web traffic
* A retailer could use it to choose a price for an item that maximizes their profit (aka "yield optimization")

#### Isotonic regression in online advertising

If you have an hour to spare, and are interested in learning more about how online advertising works - you should check out [this lecture](https://vimeo.com/137999578) that I gave in 2015 where I explain how we were able to use pair adjacent violators to solve some fun problems.

#### A picture is worth a thousand words

Here is the relationship that PAV extracts from some very noisy input data where there is an increasing relationship between _x_ and _y_:

![PAV in action](https://sanity.github.io/pairAdjacentViolators/pav-example.png)

## Features

* Smart linear interpolation between points and extrapolation outside the training data domain
* Fairly efficient implementation without compromizing code readability
* Will intelligently extrapolate to compute _y_ for values of _x_ greater or less than those used to build the PAV model

### License
Released under the [LGPL](https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License) version 3 by [Ian Clarke](http://blog.locut.us/).

