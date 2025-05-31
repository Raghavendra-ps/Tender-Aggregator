# --- START OF FILE TenFin-main/filter_engine.py ---

import re
import json
from datetime import datetime, date # Import date directly
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import logging

# Configure logging
filter_logger = logging.getLogger('filter_engine')
filter_logger.setLevel(logging.INFO)
filter_logger.propagate = False
if not filter_logger.hasHandlers():
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[%(levelname)s - FilterEngine] %(message)s"))
    filter_logger.addHandler(sh)

# --- TAG_REGEX (Includes SourceSiteKey and NEW ROT Tags) ---
TAG_REGEX = {
    # --- List Page Tags (Regular Tenders) ---
    "ListPageTenderId": re.compile(r"<ListPageTenderId>\s*(.*?)\s*</ListPageTenderId>", re.DOTALL),
    "ListPageTitle": re.compile(r"<ListPageTitle>\s*(.*?)\s*</ListPageTitle>", re.DOTALL),
    "EpublishedDateStr": re.compile(r"<EpublishedDateStr>\s*(.*?)\s*</EpublishedDateStr>", re.DOTALL),
    "ClosingDateStr": re.compile(r"<ClosingDateStr>\s*(.*?)\s*</ClosingDateStr>", re.DOTALL),
    "OpeningDateStr": re.compile(r"<OpeningDateStr>\s*(.*?)\s*</OpeningDateStr>", re.DOTALL),
    "ListPageOrgChain": re.compile(r"<ListPageOrgChain>\s*(.*?)\s*</ListPageOrgChain>", re.DOTALL),
    "DetailPageLink": re.compile(r"<DetailPageLink>\s*(.*?)\s*</DetailPageLink>", re.DOTALL),
    
    # --- Source Site Key (Common) ---
    "SourceSiteKey": re.compile(r"<SourceSiteKey>\s*(.*?)\s*</SourceSiteKey>", re.DOTALL), 
    
    # --- Detail Page - Basic (Regular Tenders) ---
    "OrganisationChain": re.compile(r"<OrganisationChain>\s*(.*?)\s*</OrganisationChain>", re.DOTALL),
    "TenderReferenceNumber": re.compile(r"<TenderReferenceNumber>\s*(.*?)\s*</TenderReferenceNumber>", re.DOTALL),
    "DetailTenderId": re.compile(r"<DetailTenderId>\s*(.*?)\s*</DetailTenderId>", re.DOTALL),
    "WithdrawalAllowed": re.compile(r"<WithdrawalAllowed>\s*(.*?)\s*</WithdrawalAllowed>", re.DOTALL),
    "TenderType": re.compile(r"<TenderType>\s*(.*?)\s*</TenderType>", re.DOTALL),
    "FormOfContract": re.compile(r"<FormOfContract>\s*(.*?)\s*</FormOfContract>", re.DOTALL),
    "TenderCategory": re.compile(r"<TenderCategory>\s*(.*?)\s*</TenderCategory>", re.DOTALL),
    "NoOfCovers": re.compile(r"<NoOfCovers>\s*(.*?)\s*</NoOfCovers>", re.DOTALL),
    "GeneralTechEvaluation": re.compile(r"<GeneralTechEvaluation>\s*(.*?)\s*</GeneralTechEvaluation>", re.DOTALL),
    "ItemwiseTechEvaluation": re.compile(r"<ItemwiseTechEvaluation>\s*(.*?)\s*</ItemwiseTechEvaluation>", re.DOTALL),
    "PaymentMode": re.compile(r"<PaymentMode>\s*(.*?)\s*</PaymentMode>", re.DOTALL),
    "MultiCurrencyBoq": re.compile(r"<MultiCurrencyBoq>\s*(.*?)\s*</MultiCurrencyBoq>", re.DOTALL),
    "MultiCurrencyFee": re.compile(r"<MultiCurrencyFee>\s*(.*?)\s*</MultiCurrencyFee>", re.DOTALL),
    "TwoStageBidding": re.compile(r"<TwoStageBidding>\s*(.*?)\s*</TwoStageBidding>", re.DOTALL),
    
    # --- Payment Instruments (Structured - Regular Tenders) ---
    "PaymentInstrumentSNo": re.compile(r"<PaymentInstrumentSNo>\s*(.*?)\s*</PaymentInstrumentSNo>", re.DOTALL),
    "PaymentInstrumentType": re.compile(r"<PaymentInstrumentType>\s*(.*?)\s*</PaymentInstrumentType>", re.DOTALL),
    
    # --- Cover Info (Structured - Regular Tenders) ---
    "CoverInfoNo": re.compile(r"<CoverInfoNo>\s*(.*?)\s*</CoverInfoNo>", re.DOTALL),
    "CoverInfoType": re.compile(r"<CoverInfoType>\s*(.*?)\s*</CoverInfoType>", re.DOTALL),
    
    # --- Detail Page - Fees (Regular Tenders) ---
    "TenderFee": re.compile(r"<TenderFee>\s*(.*?)\s*</TenderFee>", re.DOTALL),
    "ProcessingFee": re.compile(r"<ProcessingFee>\s*(.*?)\s*</ProcessingFee>", re.DOTALL),
    "FeePayableTo": re.compile(r"<FeePayableTo>\s*(.*?)\s*</FeePayableTo>", re.DOTALL),
    "FeePayableAt": re.compile(r"<FeePayableAt>\s*(.*?)\s*</FeePayableAt>", re.DOTALL),
    "TenderFeeExemption": re.compile(r"<TenderFeeExemption>\s*(.*?)\s*</TenderFeeExemption>", re.DOTALL),
    "EmdAmount": re.compile(r"<EmdAmount>\s*(.*?)\s*</EmdAmount>", re.DOTALL),
    "EmdExemption": re.compile(r"<EmdExemption>\s*(.*?)\s*</EmdExemption>", re.DOTALL),
    "EmdFeeType": re.compile(r"<EmdFeeType>\s*(.*?)\s*</EmdFeeType>", re.DOTALL),
    "EmdPercentage": re.compile(r"<EmdPercentage>\s*(.*?)\s*</EmdPercentage>", re.DOTALL),
    "EmdPayableTo": re.compile(r"<EmdPayableTo>\s*(.*?)\s*</EmdPayableTo>", re.DOTALL),
    "EmdPayableAt": re.compile(r"<EmdPayableAt>\s*(.*?)\s*</EmdPayableAt>", re.DOTALL),
    
    # --- Detail Page - Work Item (Regular Tenders) ---
    "WorkTitle": re.compile(r"<WorkTitle>\s*(.*?)\s*</WorkTitle>", re.DOTALL),
    "WorkDescription": re.compile(r"<WorkDescription>\s*(.*?)\s*</WorkDescription>", re.DOTALL),
    "NdaPreQualification": re.compile(r"<NdaPreQualification>\s*(.*?)\s*</NdaPreQualification>", re.DOTALL),
    "IndependentExternalMonitorRemarks": re.compile(r"<IndependentExternalMonitorRemarks>\s*(.*?)\s*</IndependentExternalMonitorRemarks>", re.DOTALL),
    "TenderValue": re.compile(r"<TenderValue>\s*(.*?)\s*</TenderValue>", re.DOTALL),
    "ProductCategory": re.compile(r"<ProductCategory>\s*(.*?)\s*</ProductCategory>", re.DOTALL),
    "SubCategory": re.compile(r"<SubCategory>\s*(.*?)\s*</SubCategory>", re.DOTALL),
    "ContractType": re.compile(r"<ContractType>\s*(.*?)\s*</ContractType>", re.DOTALL),
    "BidValidityDays": re.compile(r"<BidValidityDays>\s*(.*?)\s*</BidValidityDays>", re.DOTALL),
    "PeriodOfWorkDays": re.compile(r"<PeriodOfWorkDays>\s*(.*?)\s*</PeriodOfWorkDays>", re.DOTALL),
    "Location": re.compile(r"<Location>\s*(.*?)\s*</Location>", re.DOTALL),
    "Pincode": re.compile(r"<Pincode>\s*(.*?)\s*</Pincode>", re.DOTALL),
    "PreBidMeetingPlace": re.compile(r"<PreBidMeetingPlace>\s*(.*?)\s*</PreBidMeetingPlace>", re.DOTALL),
    "PreBidMeetingAddress": re.compile(r"<PreBidMeetingAddress>\s*(.*?)\s*</PreBidMeetingAddress>", re.DOTALL),
    "PreBidMeetingDate": re.compile(r"<PreBidMeetingDate>\s*(.*?)\s*</PreBidMeetingDate>", re.DOTALL),
    "BidOpeningPlace": re.compile(r"<BidOpeningPlace>\s*(.*?)\s*</BidOpeningPlace>", re.DOTALL),
    "AllowNdaTender": re.compile(r"<AllowNdaTender>\s*(.*?)\s*</AllowNdaTender>", re.DOTALL),
    "AllowPreferentialBidder": re.compile(r"<AllowPreferentialBidder>\s*(.*?)\s*</AllowPreferentialBidder>", re.DOTALL),
    
    # --- Detail Page - Critical Dates (Regular Tenders) ---
    "PublishedDate": re.compile(r"<PublishedDate>\s*(.*?)\s*</PublishedDate>", re.DOTALL),
    "DetailBidOpeningDate": re.compile(r"<DetailBidOpeningDate>\s*(.*?)\s*</DetailBidOpeningDate>", re.DOTALL),
    "DocDownloadStartDate": re.compile(r"<DocDownloadStartDate>\s*(.*?)\s*</DocDownloadStartDate>", re.DOTALL),
    "DocDownloadEndDate": re.compile(r"<DocDownloadEndDate>\s*(.*?)\s*</DocDownloadEndDate>", re.DOTALL),
    "ClarificationStartDate": re.compile(r"<ClarificationStartDate>\s*(.*?)\s*</ClarificationStartDate>", re.DOTALL),
    "ClarificationEndDate": re.compile(r"<ClarificationEndDate>\s*(.*?)\s*</ClarificationEndDate>", re.DOTALL),
    "BidSubmissionStartDate": re.compile(r"<BidSubmissionStartDate>\s*(.*?)\s*</BidSubmissionStartDate>", re.DOTALL),
    "BidSubmissionEndDate": re.compile(r"<BidSubmissionEndDate>\s*(.*?)\s*</BidSubmissionEndDate>", re.DOTALL),
    
    # --- Detail Page - TIA (Regular Tenders) ---
    "TenderInvitingAuthorityName": re.compile(r"<TenderInvitingAuthorityName>\s*(.*?)\s*</TenderInvitingAuthorityName>", re.DOTALL),
    "TenderInvitingAuthorityAddress": re.compile(r"<TenderInvitingAuthorityAddress>\s*(.*?)\s*</TenderInvitingAuthorityAddress>", re.DOTALL),
    
    # --- Detail Page - Documents (Structured - Regular Tenders) ---
    "DocumentName": re.compile(r"<DocumentName>\s*(.*?)\s*</DocumentName>", re.DOTALL),
    "DocumentLink": re.compile(r"<DocumentLink>\s*(.*?)\s*</DocumentLink>", re.DOTALL),
    "DocumentDescription": re.compile(r"<DocumentDescription>\s*(.*?)\s*</DocumentDescription>", re.DOTALL),
    "DocumentSize": re.compile(r"<DocumentSize>\s*(.*?)\s*</DocumentSize>", re.DOTALL),

    # --- ROT (Results of Tenders) Specific Tags ---
    "RotSNo": re.compile(r"<RotSNo>\s*(.*?)\s*</RotSNo>", re.DOTALL),
    "RotTenderId": re.compile(r"<RotTenderId>\s*(.*?)\s*</RotTenderId>", re.DOTALL),
    "RotTitleRef": re.compile(r"<RotTitleRef>\s*(.*?)\s*</RotTitleRef>", re.DOTALL),
    "RotOrganisationChain": re.compile(r"<RotOrganisationChain>\s*(.*?)\s*</RotOrganisationChain>", re.DOTALL),
    "RotTenderStage": re.compile(r"<RotTenderStage>\s*(.*?)\s*</RotTenderStage>", re.DOTALL),
    "RotStatusDetailPageLink": re.compile(r"<RotStatusDetailPageLink>\s*(.*?)\s*</RotStatusDetailPageLink>", re.DOTALL),
    "RotStageSummaryFileStatus": re.compile(r"<RotStageSummaryFileStatus>\s*(.*?)\s*</RotStageSummaryFileStatus>", re.DOTALL),
    "RotStageSummaryFilePath": re.compile(r"<RotStageSummaryFilePath>\s*(.*?)\s*</RotStageSummaryFilePath>", re.DOTALL),
    "RotStageSummaryFilename": re.compile(r"<RotStageSummaryFilename>\s*(.*?)\s*</RotStageSummaryFilename>", re.DOTALL),
}


def parse_tender_blocks_from_tagged_file(file_path: Path) -> List[str]:
    """Parses a file containing multiple tender blocks separated by standard markers."""
    if not file_path.is_file(): filter_logger.error(f"Source file not found: {file_path}"); return []
    try: content = file_path.read_text(encoding="utf-8", errors='replace')
    except Exception as e: filter_logger.error(f"Read failed for {file_path}: {e}"); return []
    raw_blocks = [b for b in content.split("--- TENDER START ---") if b.strip()]
    processed_blocks = [re.sub(r"--- TENDER END ---.*", "", b, flags=re.DOTALL).strip() for b in raw_blocks]
    filter_logger.debug(f"Split {len(processed_blocks)} blocks from {file_path.name}")
    return [b for b in processed_blocks if b]


def extract_tender_info_from_tagged_block(block_text: str) -> Dict[str, Any]:
    """
    Extracts tender information from a single tagged block.
    Detects if it's a regular tender or an ROT tender based on unique tags.
    """
    
    # Helper to convert CamelCase to snake_case
    def to_snake_case(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    # --- Determine if it's an ROT block by checking for a unique ROT tag ---
    is_rot_block = False
    if TAG_REGEX["RotSNo"].search(block_text): # RotSNo is a good unique indicator
        is_rot_block = True
        
    # Initialize tender dictionary based on type
    if is_rot_block:
        tender: Dict[str, Any] = {
            "rot_s_no": "N/A", "rot_tender_id": "N/A", "rot_title_ref": "N/A",
            "rot_organisation_chain": "N/A", "rot_tender_stage": "N/A",
            "rot_status_detail_page_link": "N/A", "rot_stage_summary_file_status": "N/A",
            "rot_stage_summary_file_path": "N/A", "rot_stage_summary_filename": "N/A",
            "source_site_key": "N/A",
            # Primary fields for ROT
            "primary_tender_id": "N/A", "primary_title": "N/A",
            "data_type": "rot" # Add a type identifier
        }
        # Define which tags are single-value for ROT
        single_value_tag_keys_rot = [
            "RotSNo", "RotTenderId", "RotTitleRef", "RotOrganisationChain",
            "RotTenderStage", "RotStatusDetailPageLink", "RotStageSummaryFileStatus",
            "RotStageSummaryFilePath", "RotStageSummaryFilename", "SourceSiteKey"
        ]
        tags_to_process = single_value_tag_keys_rot
        structured_tag_groups_to_process = {} # No structured lists for ROT currently

    else: # Regular tender block
        tender = {
            "list_page_tender_id": "N/A", "list_page_title": "N/A", "epublished_date_str": "N/A",
            "closing_date_str": "N/A", "opening_date_str": "N/A", "list_page_org_chain": "N/A",
            "detail_page_link": "N/A", "source_site_key": "N/A", 
            "organisation_chain": "N/A", "tender_reference_number": "N/A", "detail_tender_id": "N/A",
            "withdrawal_allowed": "N/A", "tender_type": "N/A", "form_of_contract": "N/A",
            "tender_category": "N/A", "no_of_covers": "N/A", "general_tech_evaluation": "N/A",
            "itemwise_tech_evaluation": "N/A", "payment_mode": "N/A", "multi_currency_boq": "N/A",
            "multi_currency_fee": "N/A", "two_stage_bidding": "N/A", 
            "tender_fee": "N/A", "processing_fee": "N/A",
            "fee_payable_to": "N/A", "fee_payable_at": "N/A",
            "tender_fee_exemption": "N/A", "emd_amount": "N/A", "emd_exemption": "N/A",
            "emd_fee_type": "N/A", "emd_percentage": "N/A", "emd_payable_to": "N/A",
            "emd_payable_at": "N/A", "work_title": "N/A", "work_description": "N/A",
            "nda_pre_qualification": "N/A", 
            "independent_external_monitor_remarks": "N/A",
            "tender_value": "N/A", "product_category": "N/A",
            "sub_category": "N/A", "contract_type": "N/A", "bid_validity_days": "N/A",
            "period_of_work_days": "N/A", "location": "N/A", "pincode": "N/A",
            "pre_bid_meeting_place": "N/A", "pre_bid_meeting_address": "N/A", "pre_bid_meeting_date": "N/A",
            "bid_opening_place": "N/A", "allow_nda_tender": "N/A", "allow_preferential_bidder": "N/A",
            "published_date": "N/A", "detail_bid_opening_date": "N/A", "doc_download_start_date": "N/A",
            "doc_download_end_date": "N/A", "clarification_start_date": "N/A", "clarification_end_date": "N/A",
            "bid_submission_start_date": "N/A", "bid_submission_end_date": "N/A",
            "tender_inviting_authority_name": "N/A", "tender_inviting_authority_address": "N/A",
            "payment_instruments": [], "covers_info": [], "tender_documents": [],
            # Primary fields for regular
            "primary_tender_id": "N/A", "primary_title": "N/A",
            "primary_published_date": "N/A", "primary_closing_date": "N/A", "primary_opening_date": "N/A",
            "data_type": "regular" # Add a type identifier
        }
        single_value_tag_keys_regular = [
            tag for tag in TAG_REGEX
            if tag not in [ 
                "PaymentInstrumentSNo", "PaymentInstrumentType",
                "CoverInfoNo", "CoverInfoType",
                "DocumentName", "DocumentLink", "DocumentDescription", "DocumentSize",
                # Exclude ROT tags here explicitly if TAG_REGEX becomes too large
                "RotSNo", "RotTenderId", "RotTitleRef", "RotOrganisationChain",
                "RotTenderStage", "RotStatusDetailPageLink", "RotStageSummaryFileStatus",
                "RotStageSummaryFilePath", "RotStageSummaryFilename"
            ]
        ]
        tags_to_process = single_value_tag_keys_regular
        structured_tag_groups_to_process = {
            "payment_instruments": {
                "list_key": "payment_instruments",
                "item_keys_map": {"s_no": "PaymentInstrumentSNo", "instrument_type": "PaymentInstrumentType"}
            },
            "covers_info": {
                "list_key": "covers_info",
                "item_keys_map": {"cover_no": "CoverInfoNo", "cover_type": "CoverInfoType"}
            },
            "tender_documents": {
                "list_key": "tender_documents",
                "item_keys_map": {
                    "name": "DocumentName", "link": "DocumentLink", 
                    "description": "DocumentDescription", "size": "DocumentSize"
                },
                "required_any": ["name", "link"] # At least name or link must be present
            }
        }

    # --- Process single-value tags ---
    for tag_key in tags_to_process:
        try:
            regex_pattern = TAG_REGEX.get(tag_key)
            if not regex_pattern:
                filter_logger.warning(f"Regex pattern missing for single tag: {tag_key} (Block type: {'ROT' if is_rot_block else 'Regular'})")
                continue
            
            match = regex_pattern.search(block_text)
            if match:
                dict_key = to_snake_case(tag_key)
                if dict_key in tender: 
                    tender[dict_key] = match.group(1).strip()
                else:
                    filter_logger.warning(f"Derived snake_case key '{dict_key}' not in initial tender dict for tag '{tag_key}'. Block type: {'ROT' if is_rot_block else 'Regular'}")
        except Exception as e: filter_logger.warning(f"Error extracting single tag '{tag_key}': {e}")

    # --- Process structured list tags (only for regular tenders) ---
    if not is_rot_block:
        for group_name, group_config in structured_tag_groups_to_process.items():
            list_key_in_tender = group_config["list_key"]
            item_keys_map = group_config["item_keys_map"]
            required_any_fields = group_config.get("required_any")

            try:
                # Find all occurrences for each field in the group
                field_values_all = {}
                min_length = float('inf')
                all_fields_present_in_regex = True

                for item_dict_key, regex_tag_key in item_keys_map.items():
                    if regex_tag_key not in TAG_REGEX:
                        filter_logger.warning(f"Regex pattern missing for structured tag: {regex_tag_key} in group {group_name}")
                        all_fields_present_in_regex = False; break
                    field_values_all[item_dict_key] = TAG_REGEX[regex_tag_key].findall(block_text)
                    min_length = min(min_length, len(field_values_all[item_dict_key]))
                
                if not all_fields_present_in_regex: continue

                num_items = min_length
                if num_items == float('inf'): num_items = 0 # Case where no fields had any matches

                # Basic check for length consistency if num_items > 0
                lengths_consistent = True
                if num_items > 0:
                    for item_dict_key in item_keys_map.keys():
                        if len(field_values_all[item_dict_key]) != num_items:
                            lengths_consistent = False
                            filter_logger.warning(f"Mismatch in {group_name} tag counts. Expected {num_items} for all, but '{item_dict_key}' has {len(field_values_all[item_dict_key])}. Data might be incomplete.")
                            break
                
                # If lengths are inconsistent, we might take the minimum or skip. For now, take min.
                # A more robust approach might be to pair based on proximity if one tag is missing for an item.

                current_list_items = []
                if num_items > 0:
                    for i in range(num_items):
                        item_dict = {}
                        valid_item = True
                        if required_any_fields: valid_item = False # Assume invalid until a required field is found

                        for item_dict_key, regex_tag_key in item_keys_map.items():
                            val = field_values_all[item_dict_key][i].strip() if i < len(field_values_all[item_dict_key]) else "N/A"
                            item_dict[item_dict_key] = val
                            if required_any_fields and item_dict_key in required_any_fields and val and val != "N/A":
                                valid_item = True
                        
                        if valid_item:
                            current_list_items.append(item_dict)
                
                tender[list_key_in_tender] = current_list_items

            except Exception as e: filter_logger.warning(f"Error extracting structured {group_name} info: {e}")
            
    # --- Set primary fields ---
    if is_rot_block:
        tender["primary_tender_id"] = tender["rot_tender_id"]
        tender["primary_title"] = tender["rot_title_ref"]
        # ROT tenders don't have these date fields directly from list pages
        tender["primary_published_date"] = "N/A" 
        tender["primary_closing_date"] = "N/A"
        tender["primary_opening_date"] = "N/A"
    else: # Regular tender
        tender["primary_tender_id"] = tender["detail_tender_id"] if tender["detail_tender_id"] != "N/A" else tender["list_page_tender_id"]
        tender["primary_title"] = tender["work_title"] if tender["work_title"] != "N/A" else tender["list_page_title"]
        tender["primary_published_date"] = tender["published_date"] if tender["published_date"] != "N/A" else tender["epublished_date_str"]
        tender["primary_closing_date"] = tender["bid_submission_end_date"] if tender["bid_submission_end_date"] != "N/A" else tender["closing_date_str"]
        tender["primary_opening_date"] = tender["detail_bid_opening_date"] if tender["detail_bid_opening_date"] != "N/A" else tender["opening_date_str"]

    return tender


def matches_filters(
    tender: Dict[str, Any], 
    keywords: List[str], 
    use_regex: bool, 
    site_key_filter: Optional[str], 
    start_date_str: Optional[str], 
    end_date_str: Optional[str]
) -> bool:
    """Checks if a tender dictionary matches the provided filter criteria."""
    
    data_type = tender.get("data_type", "regular") # Get data_type from tender dict

    tender_site_key = tender.get("source_site_key", "N/A")
    if site_key_filter and site_key_filter.lower() != tender_site_key.lower():
        return False

    # Date filtering only applies to 'regular' tenders as ROT data lacks these dates
    if data_type == "regular":
        possible_date_formats = ["%d-%b-%Y %I:%M %p", "%d-%b-%Y"] 
        filter_date_format = "%Y-%m-%d" 
        tender_opening_date: Optional[date] = None 
        tender_opening_date_str = tender.get("primary_opening_date", "N/A") 

        if tender_opening_date_str and tender_opening_date_str != "N/A":
            for fmt in possible_date_formats:
                try:
                    tender_opening_date = datetime.strptime(tender_opening_date_str, fmt).date()
                    break 
                except ValueError:
                    continue 
            if not tender_opening_date:
                 filter_logger.warning(f"Could not parse tender Opening Date '{tender_opening_date_str}' (Regular Tender). Skipping date filters for ID '{tender.get('primary_tender_id', 'N/A')}'.")

        if tender_opening_date:
            try:
                if start_date_str:
                    filter_start_date = datetime.strptime(start_date_str, filter_date_format).date()
                    if tender_opening_date < filter_start_date: return False 
                if end_date_str:
                    filter_end_date = datetime.strptime(end_date_str, filter_date_format).date()
                    if tender_opening_date > filter_end_date: return False 
            except ValueError as e_date: 
                filter_logger.error(f"Error parsing filter dates ('{start_date_str}', '{end_date_str}'): {e_date}. Skipping date filters.")
    elif data_type == "rot" and (start_date_str or end_date_str):
        filter_logger.debug(f"Date filters specified but tender type is ROT (ID: {tender.get('primary_tender_id', 'N/A')}). ROT data does not have comparable date fields. Date filter ignored for this item.")

    # --- Define search fields based on data_type ---
    if data_type == "rot":
        search_fields = [
            "primary_tender_id", # from rot_tender_id
            "primary_title",     # from rot_title_ref
            "rot_organisation_chain",
            "rot_tender_stage",
            "rot_stage_summary_filename",
            "source_site_key"
        ]
        search_content_parts = [str(tender.get(k, "")) for k in search_fields if tender.get(k) and tender.get(k) != "N/A"]
    
    else: # Regular tender
        search_fields = [
            "primary_tender_id", "primary_title", "organisation_chain", "list_page_org_chain",
            "source_site_key", 
            "work_description", "tender_reference_number", "product_category", "sub_category",
            "location", "pincode", "tender_inviting_authority_name", "fee_payable_to", "emd_payable_to",
            "bid_opening_place", "pre_bid_meeting_place", "fee_payable_at", "emd_payable_at",
            "nda_pre_qualification", "independent_external_monitor_remarks"
        ]
        search_content_parts = [str(tender.get(k, "")) for k in search_fields if tender.get(k) and tender.get(k) != "N/A"]
        
        try: 
            if tender.get("payment_instruments"):
                for pi in tender["payment_instruments"]:
                     if isinstance(pi, dict) and pi.get("instrument_type"): search_content_parts.append(str(pi["instrument_type"]))
        except Exception as e: filter_logger.warning(f"Error adding payment instruments (Regular Tender) to search: {e}")

        try: 
            if tender.get("covers_info"):
                for ci in tender["covers_info"]:
                     if isinstance(ci, dict) and ci.get("cover_type"): search_content_parts.append(str(ci["cover_type"]))
        except Exception as e: filter_logger.warning(f"Error adding cover info (Regular Tender) to search: {e}")
            
        try: 
            if tender.get("tender_documents"):
                for doc in tender["tender_documents"]:
                     if isinstance(doc, dict):
                          if doc.get("name"): search_content_parts.append(str(doc["name"]))
                          if doc.get("description"): search_content_parts.append(str(doc["description"]))
        except Exception as e: filter_logger.warning(f"Error adding document info (Regular Tender) to search: {e}")

    search_content = " ".join(filter(None, search_content_parts))

    if keywords:
        if not search_content: return False 
        search_content_lower = search_content.lower() 
        match_found = False
        try:
            if use_regex:
                if any(re.search(kw, search_content, re.IGNORECASE) for kw in keywords): match_found = True
            else:
                if any(kw.lower() in search_content_lower for kw in keywords): match_found = True
            if not match_found: return False 
        except re.error as e_re:
            filter_logger.error(f"Invalid regex: {e_re}. Skipping keyword filter for tender ID {tender.get('primary_tender_id', 'N/A')}."); 
            pass # Treat as no match on error, or simply pass if other criteria must still hold true
    return True


def run_filter(
    base_folder: Path, 
    # tender_filename: str, # This will be determined by data_type
    keywords: list, 
    use_regex: bool, 
    filter_name: str, 
    site_key: Optional[str], 
    start_date: Optional[str], 
    end_date: Optional[str],
    data_type: str = "regular" # New parameter: "regular" or "rot"
) -> str:
    filter_logger.info(f"--- Running Filter (Type: {data_type.upper()}) ---")
    filter_logger.info(f"  Filter Name : {filter_name}")
    filter_logger.info(f"  Keywords    : {keywords} (Regex: {use_regex})")
    filter_logger.info(f"  Site Key    : {site_key or 'Any'}") 
    if data_type == "regular": # Date filtering only applicable for regular
        filter_logger.info(f"  Date Range (Opening Date): {start_date or 'N/A'} to {end_date or 'N/A'}")
    else:
        filter_logger.info(f"  Date Range: Not applicable for ROT data type.")
    filter_logger.info("-----------------------------------------------------------------")

    # Determine source file pattern based on data_type
    if data_type == "rot":
        source_file_pattern = "Merged_ROT_*.txt"
        output_subdir_prefix = f"{filter_name}_ROT_Tenders"
    else: # Default to regular
        source_file_pattern = "Final_Tender_List_*.txt"
        output_subdir_prefix = f"{filter_name}_Tenders"

    # Find the latest source file matching the pattern
    source_files = sorted(
        [p for p in base_folder.glob(source_file_pattern) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not source_files:
        filter_logger.error(f"No source files matching pattern '{source_file_pattern}' found in {base_folder}. Cannot run filter.")
        raise FileNotFoundError(f"No source data file found for type '{data_type}' in {base_folder}.")
    
    latest_source_filename = source_files[0].name
    filter_logger.info(f"  Source File : {latest_source_filename} (Type: {data_type})")
    tender_path = base_folder / latest_source_filename
    
    tagged_blocks = parse_tender_blocks_from_tagged_file(tender_path)

    if not tagged_blocks:
        filter_logger.warning("No tender blocks parsed from source file. Result will be empty.")
        all_matching_tenders = []
    else:
        all_matching_tenders: List[Dict[str, Any]] = []
        processed_count = 0; match_count = 0
        filter_logger.info(f"Processing {len(tagged_blocks)} tender blocks...")
        for block_text in tagged_blocks:
            processed_count += 1
            try:
                tender_info = extract_tender_info_from_tagged_block(block_text)
                # Ensure the extracted block's data_type matches the filter's data_type
                # This is a safeguard in case a merged file contains mixed types (should not happen with current design)
                if tender_info.get("data_type") != data_type:
                    filter_logger.debug(f"Skipping block {processed_count}: data_type mismatch (expected {data_type}, got {tender_info.get('data_type')}). Tender ID: {tender_info.get('primary_tender_id','N/A')}")
                    continue

                if matches_filters(tender_info, keywords, use_regex, site_key, start_date, end_date):
                    all_matching_tenders.append(tender_info)
                    match_count += 1
            except Exception as e:
                 filter_logger.error(f"Error processing block #{processed_count}: {e}", exc_info=True)
        filter_logger.info(f"Initial processing complete. Found {match_count} matching tenders (before deduplication).")

    unique_tender_dictionaries: List[Dict[str, Any]] = []
    seen_tender_ids: Set[str] = set()
    duplicates_removed = 0
    for tender in all_matching_tenders:
        tender_id_val = tender.get("primary_tender_id") # This is now standardized
        if tender_id_val and tender_id_val != "N/A":
            if tender_id_val not in seen_tender_ids:
                unique_tender_dictionaries.append(tender)
                seen_tender_ids.add(tender_id_val)
            else: duplicates_removed += 1
        else: # Tender with no valid ID - include it but log a warning
            unique_tender_dictionaries.append(tender)
            filter_logger.warning(f"Tender included with missing/invalid primary_tender_id: Title '{tender.get('primary_title', 'N/A')}' (Type: {tender.get('data_type', 'unknown')})")
    
    filter_logger.info(f"Removed {duplicates_removed} duplicate tenders based on primary_tender_id.")
    filter_logger.info(f"Total unique tenders to save: {len(unique_tender_dictionaries)}")

    safe_filter_name = re.sub(r'[^\w\s-]', '', filter_name).strip()
    safe_filter_name = re.sub(r'[-\s]+', '_', safe_filter_name) if safe_filter_name else "UnnamedFilter"
    
    # Adjust output folder based on data_type
    output_folder_base_name = "Filtered Tenders"
    if data_type == "rot":
        output_folder_base_name = "Filtered Tenders ROT"

    output_folder = base_folder / output_folder_base_name / f"{safe_filter_name}_Results" # Use general "Results"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    output_filename = "Filtered_Tenders.json" # Keep standard filename within folder
    output_path = output_folder / output_filename
    try:
        filter_logger.info(f"Saving {len(unique_tender_dictionaries)} unique tender dictionaries to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(unique_tender_dictionaries, f, indent=2, ensure_ascii=False, default=str) 
        filter_logger.info("Save successful.")
    except Exception as e:
        filter_logger.error(f"Failed to write output JSON to {output_path}: {e}")
        raise IOError(f"Failed to write output JSON: {e}") from e

    return str(output_path.resolve())

# --- END OF FILE TenFin-main/filter_engine.py ---
