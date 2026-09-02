<?php

declare(strict_types=1);

namespace Vegoia\Interop;

use function class_exists;

use Fhaculty\Graph\Edge\Directed;
use Fhaculty\Graph\Graph as ExternalGraph;

/**
 * Reads a graph built with graphp/graph, the established PHP graph library.
 *
 * That library models a graph as objects -- a Vertex per node, an Edge per
 * connection, each with its own attributes -- which is a good way to build and
 * edit one and the wrong way to sweep it a million times. This class is the
 * bridge: keep using it for the parts of your program that manipulate the
 * graph, hand it here for the parts that measure it.
 *
 * It is a development dependency only, and stays one. Nothing in this library
 * requires it at runtime, isAvailable() answers whether it is installed
 * without loading anything, and a program that never calls import() never
 * touches it.
 *
 * Two things it allows that this library does not, and both are resolved here
 * rather than silently:
 *
 * A graphp graph may mix directed and undirected edges. Vegoia's graph is one
 * or the other, because modularity, betweenness and the rest are different
 * formulas in the two cases rather than the same formula with a flag. A mixed
 * graph is therefore read as directed, with each undirected edge becoming a
 * pair of arrows -- which is what "undirected" means to a directed algorithm,
 * and is stated in the return rather than assumed.
 *
 * An edge may have no weight at all, which graphp reports as null. That is
 * read as 1.0: an edge that exists but says nothing about how much.
 */
final class Graphp
{
    /** Whether graphp/graph is installed, without loading any of it. */
    public static function isAvailable(): bool
    {
        return class_exists(ExternalGraph::class, autoload: true);
    }

    /**
     * No installed-or-not check here: the parameter is typed as a graphp
     * graph, so a caller who reached this method already holds one, and a
     * caller who cannot build one cannot call it. isAvailable() is for asking
     * beforehand.
     */
    public static function import(ExternalGraph $graph): LabelledGraph
    {
        $nodes = [];

        foreach ($graph->getVertices()->getVector() as $vertex) {
            $nodes[] = (string) $vertex->getId();
        }

        $directed = false;

        foreach ($graph->getEdges()->getVector() as $edge) {
            if ($edge instanceof Directed) {
                $directed = true;

                break;
            }
        }

        $edges = [];

        foreach ($graph->getEdges()->getVector() as $edge) {
            $vertices = $edge->getVertices()->getVector();
            $from = (string) $vertices[0]->getId();
            $to = (string) $vertices[1]->getId();

            // graphp reports an unweighted edge as null rather than 1.
            $weight = $edge->getWeight();
            $weight = $weight === null ? 1.0 : (float) $weight;

            if ($edge instanceof Directed) {
                $start = $edge->getVerticesStart()->getVector()[0];
                $target = $edge->getVerticesTarget()->getVector()[0];

                $edges[] = [(string) $start->getId(), (string) $target->getId(), $weight];

                continue;
            }

            $edges[] = [$from, $to, $weight];

            // In a graph that also has arrows, an undirected edge is both.
            if ($directed && $from !== $to) {
                $edges[] = [$to, $from, $weight];
            }
        }

        return LabelledGraph::fromEdges($edges, $nodes, $directed);
    }
}
