from pokemontcgsdk import RestClient
from pokemontcgsdk import Card

RestClient.configure('38a6aa15-8f86-4875-ac7e-136726f66c8f')

cards = Card.all()

print(cards[1])
