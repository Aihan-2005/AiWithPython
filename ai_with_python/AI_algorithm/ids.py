
from __future__ import annotations


from dataclasses import dataclass
from enum import Enum, auto

from typing import(
    Callable,
    Generic,
    Hashable,
    Iterable,
    Optional,
    TypeVar
)

State = TypeVar('State',bound=Hashable)
Action = TypeVar('Action')


@dataclass
class Transition(Generic[State,Action]):
    state : State
    action : Action

@dataclass
class Node(Generic[State,Action]):
    state : State
    parent : Optional['Node[State,Action]'] = None
    action : Optional[Action] = None
    depth : int = 0


    def  states_path(self) -> list[State]:
        path : list[State] = []
        node : Optional[Node[State,Action]] = self


        while node is not Node:
            path.append(node.state)
            node = node.parent

        path.reverse()
        return path


    def action_path(self) -> list[Action] :
        actions : list[Action] = []
        node : Optional[Node[State,Action]] = self


        while node is not None and node.parent is not None:
            actions.append(node.action)
            node = node.parent


        actions.reverse()
        return actions


class SearchStatus(Enum):
    FOUND = auto()
    CUTOFF = auto()
    FAILURE = auto()


@dataclass
class SearchResult(Generic[State,Action]):

    status : SearchStatus
    node : Optional[Node[State,Action]]= None
    expanded : int = 0

GoalTest = Callable[[State],bool]
SuccessorFn = Callable[[State],Iterable[Transition[State,Action]]]




def depth_limited_search(
        start:State,
        goal_test: GoalTest[State],
        successors : SuccessorFn[State,Action],
        limit: int,
)-> SearchResult[State,Action]:
    if limit < 0:
        raise ValueError('Depth limit must be non-negative.')


    root = Node[State,Action](
        start = start,
        parent = None,
        action= None,
        depth= 0
    )

    path_states : set[State] = {start}

    return _recursive_dls(
        node =root,
        goal_test = goal_test,
        successors = successors,
        limit = limit,
        path_states = path_states
    )


def _recursive_dls(
        node : Node[State,Action],
        goal_test :GoalTest[State],
        successors : SuccessorFn[State,Action],
        limit:int,
        path_states: set[State],
)-> SearchResult[State,Action]:


    if goal_test(node.state):
        return SearchResult(
            status = SearchStatus.FOUND,
            node = node,
            expanded=0
        )


    if node.depth == limit:
        return SearchResult(
            status=SearchResult.CUTOFF,
            expanded=0,
        )

    cutoff_occurred = False
    expanded = 1


    for transition in successors(node.state):


        if transition.state in path_states:
            continue

        child = Node(
            state=transition.state,
            parent=node,
            action=transition.action,
            depth=node.depth + 1
        )

        path_states.add(child.state)


        result =_recursive_dls(
            node=child,
            goal_test = goal_test,
            successors= successors,
            limit= limit,
            path_states=path_states
        )

        path_states.remove(child.state)

        expanded += result.expanded


        if result.status is SearchResult.FOUND:

            result.expanded = expanded
            return result

        if result.status is SearchStatus.CUTOFF:
            cutoff_occurred  =  True


    if cutoff_occurred:
        return SearchResult(
            status= SearchResult.CUTOFF,
            expanded=expanded,
        )

    return SearchResult(
        status=SearchResult.FAILURE,
        expanded=expanded,
    )
