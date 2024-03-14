from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

def voting_hard(args):
    """
    Creates a VotingClassifier with hard voting strategy.
    
    Args:
    classifiers_list (list): List of tuples containing (classifier_name, classifier_instance).
    
    Returns:
    VotingClassifier: A VotingClassifier instance with hard voting strategy.
    """
    return VotingClassifier(estimators=args, voting='hard')

def stacking(args):
    """
    Creates a StackingClassifier.
    
    Args:
    classifiers_list (list): List of tuples containing (classifier_name, classifier_instance).
    
    Returns:
    StackingClassifier: A StackingClassifier instance.
    """
    return StackingClassifier(estimators=args)
