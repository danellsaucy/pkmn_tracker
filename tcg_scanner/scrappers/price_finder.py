import datetime
import os
from typing import List, Tuple, Optional
import cv2
import numpy as np
from detector_v2 import PokemonCardScanner  # Import your scanner class
from pkmncards_com import PkmnCardFinder
from google_search import GoogleSearchScraper
from scrape_tcgplayer import TcgplayerProductFetcher
from database_layer import DatabaseFactory, CardRecord
from database_config import (
    get_active_db_config, 
    CARD_SCANNER_CONFIG, 
    APP_SETTINGS,
    get_log_path,
    get_export_path
)


class CardSearcher:
    def __init__(self, config_file: str, checkpoint_file: str, hash_dict_path: str):
        """
        Initialize the Card Searcher with Pokemon Card Scanner
        
        Args:
            config_file: Path to model config file
            checkpoint_file: Path to model checkpoint
            hash_dict_path: Path to JSON file containing card hashes
        """
        self.scanner = PokemonCardScanner(config_file, checkpoint_file, hash_dict_path)
        self.db = DatabaseFactory.create_from_config(get_active_db_config())
        # Initialize your original CardSearcher
        # Uncomment and modify when ready to integrate:
    
    def save_warped_image_temp(self, warped_image: np.ndarray, temp_path: str = "temp_card.jpg") -> str:
        """Save warped image temporarily for uploading"""
        cv2.imwrite(temp_path, warped_image)
        return temp_path

    
    def get_reference_card_path(self, card_name: str, base_path: str = "downloaded_cards") -> Optional[str]:
        """
        Get the path to the reference card image
        
        Args:
            card_name: Name of the card from matches
            base_path: Base directory where card images are stored
            
        Returns:
            Path to the reference card image or None if not found
        """
        # Common image extensions to try
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        
        for ext in extensions:
            card_path = os.path.join(base_path, f"{card_name}{ext}")
            if os.path.exists(card_path):
                return card_path
        
        # Also try without extension in case it's already included
        card_path = os.path.join(base_path, card_name)
        if os.path.exists(card_path):
            return card_path
            
        return None
    
    def _should_update_price(self, card: CardRecord) -> bool:
        """Check if card price should be updated based on age (24 hours)"""
        if not card.last_checked:
            return True  # Never checked = always refresh
        
        hours_since_check = (datetime.datetime.now() - card.last_checked).total_seconds() / 3600
        needs_update = hours_since_check >= 24
        
        return needs_update

    def search_card(self, 
                   image_path: str, 
                   output_path: Optional[str] = None,
                   top_k: int = 5,
                   max_urls: int = 10,
                   show_debug: bool = False,
                   headless: bool = True,
                   use_reference_image: bool = True,
                   save_to_db: bool = True,
                   reference_cards_path: str = "downloaded_cards") -> Tuple[List[Tuple[str, int]], List[str]]:
        """
        Complete workflow: scan card and perform reverse image search
        
        Args:
            image_path: Path to input image
            output_path: Path to save warped card (optional)
            top_k: Number of top card matches to return
            max_urls: Maximum number of URLs to return
            show_debug: Whether to show debug images
            headless: Whether to run browser in headless mode
            use_reference_image: Whether to use reference card image for search (recommended)
            reference_cards_path: Path to folder containing reference card images
            
        Returns:
            Tuple of (card_matches, tcgplayer_urls)
        """
        print("🔍 Scanning Pokemon card...")
        
        # Step 1: Scan and identify the card
        try:
            warped_card, matches = self.scanner.scan_card(
                image_path=image_path,
                output_path=output_path,
                top_k=top_k,
                show_debug=show_debug
            )

            print(f"✅ Card scanning complete. Found {len(matches)} matches.")

            if not matches:
                print("❌ No matches found.")
                return [], []

            print("\n🃏 Top card matches:")
            for i, (card_name, distance) in enumerate(matches, 1):
                print(f"  {i}. {card_name} (distance={distance})")

            if matches[0][1] < 10:
                best_match_name = matches[0][0]
                print(f"\n✅ Auto-selected best match: {best_match_name} (distance={matches[0][1]})")
            else:
                print("\n⚠️ Top match is not strong enough. Please choose the correct card:")
                for i, (card_name, distance) in enumerate(matches, 1):
                    print(f"  {i}. {card_name} (distance={distance})")
                print(matches)
                while True:
                    try:
                        choice = int(input("Enter the number of the correct card (1-N): "))
                        if 1 <= choice <= len(matches):
                            best_match_name = matches[choice - 1][0]
                            break
                        else:
                            print("❌ Invalid selection. Try again.")
                    except ValueError:
                        print("❌ Please enter a valid number.")

        except Exception as e:
            print(f"❌ Error scanning card: {e}")
            return [], []
        
        
        # Use PokemonCards.com to find title
        reference_path = self.get_reference_card_path(best_match_name, reference_cards_path)
        set, card_name = best_match_name.split("/")
        card_name = card_name.removesuffix(".jpg")
        needs_update = False
        print(card_name)
        if save_to_db:
            existing_card = self.db.get_card_by_name(card_name)
            if existing_card:
                # Update existing card
                needs_update = self._should_update_price(existing_card)

            if needs_update or not existing_card:
                finder = PkmnCardFinder(set_name=set, card_img_id=card_name)
                card_title = finder.fetch_title()
                
                search_tcg = GoogleSearchScraper(query=card_title + ' tcgplayer1', target_site="tcgplayer.com")
                product_id = search_tcg.get_first_match()
                product_id = product_id.split("/product/")[1].split("/")[0]

                fetcher = TcgplayerProductFetcher(product_id=product_id)
                price = fetcher.print_market_price()
                if needs_update:
                    last_checked = datetime.datetime.now()
                    self.db.update_card_price(existing_card.id, price, last_checked)
                else:
                    last_checked = datetime.datetime.now()
                    new_card = CardRecord(
                        card_name=card_name,
                        image_path=os.path.abspath(reference_path),
                        last_price=price,
                        last_checked=last_checked
                    )
                    
                    card_id = self.db.insert_card(new_card)

            else:
                price = existing_card.last_price
                card_name = existing_card.card_name
                last_checked = existing_card.last_checked


        return {
            'card_name': card_name,
            'price': price,
            'last_checked': last_checked
        }
                
    
    # def __del__(self):
    #     """Cleanup driver on object destruction"""
    #     if self.driver:
    #         self.driver.quit()


def search_pokemon_card(image_path: str,
                       config_file: str,
                       checkpoint_file: str,
                       hash_dict_path: str,
                       output_path: Optional[str] = None,
                       top_k: int = 5,
                       max_urls: int = 10,
                       use_reference_image: bool = True,
                       reference_cards_path: str = "downloaded_cards") -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Convenience function to scan and search a Pokemon card
    
    Args:
        image_path: Path to input image
        config_file: Path to model config
        checkpoint_file: Path to model checkpoint
        hash_dict_path: Path to hash dictionary JSON
        output_path: Path to save warped card (optional)
        top_k: Number of top card matches to return
        max_urls: Maximum number of URLs to return
        use_reference_image: Whether to use reference card image for search
        reference_cards_path: Path to folder containing reference card images
        
    Returns:
        Tuple of (card_matches, tcgplayer_urls)
    """
    searcher = CardSearcher(config_file, checkpoint_file, hash_dict_path)
    return searcher.search_card(
        image_path, output_path, top_k, max_urls, 
        use_reference_image=use_reference_image,
        reference_cards_path=reference_cards_path
    )


# Example usage
if __name__ == "__main__":
    # Configuration paths
    config_file = r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\my_config.py'
    checkpoint_file = r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\best_coco_segm_mAP_epoch_99.pth'
    hash_dict_path = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\card_hashes.json"
    image_path = r"C:\Users\daforbes\Downloads\s-l1600 (1).jpg"
    
    # Create searcher instance
    searcher = CardSearcher(
            config_file=CARD_SCANNER_CONFIG['config_file'],
            checkpoint_file=CARD_SCANNER_CONFIG['checkpoint_file'],
            hash_dict_path=CARD_SCANNER_CONFIG['hash_dict_path']
        )
    
    # Perform complete search using reference card image
    result = searcher.search_card(
        image_path=image_path,
        output_path="warped_card_for_search.png",
        top_k=5,
        max_urls=10,
        show_debug=False,
        headless=True,  # Set to False to see browser in action
        use_reference_image=True,  # Use clean reference image from database
        reference_cards_path="downloaded_cards"  # Path to your card images folder
    )
    print(result)
    # print("CARD FOUND IS: ", result.card_title)
    # print("PRICE: ", result.price)