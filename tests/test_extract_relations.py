"""Regression tests for src/extract_relations.py.

Each negative case is a failure mode found in the audit of the first version
(data/extracted_relations_audit_provisional.csv); each positive case guards against over-blocking.
Run from the cti_graph_rag directory:  python -m unittest tests.test_extract_relations -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from extract_relations import (event_time, extract_scope, find_assertions, find_relations,  # noqa: E402
                               parse_date_mention)


def segment(text, *mentions):
    """Build a segment from text and (surface, type) pairs; each surface is located in order."""
    entities, pos = [], 0
    for surface, etype in mentions:
        start = text.index(surface, pos)
        entities.append({"text": surface, "type": etype, "char_start": start, "char_end": start + len(surface)})
        pos = start + len(surface)
    return {"text": text, "entities": entities}


def relations(text, *mentions):
    return {(rel, h["text"], t["text"]) for rel, h, t, *_ in find_relations(segment(text, *mentions), {})}


def confidences(text, *mentions):
    return {(rel, h["text"], t["text"]): conf for rel, h, t, _, _, conf in find_relations(segment(text, *mentions), {})}


class ValidRelationsStillFound(unittest.TestCase):
    def test_active_uses(self):
        self.assertIn(("USES", "APT29", "Cobalt Strike"),
                      relations("APT29 frequently uses Cobalt Strike.", ("APT29", "GROUP"), ("Cobalt Strike", "MALWARE")))

    def test_passive_by_reverses_direction(self):
        found = relations("Trickbot was distributed by the Emotet service.",
                          ("Trickbot", "MALWARE"), ("Emotet", "MALWARE"))
        self.assertEqual(found, {("USES", "Emotet", "Trickbot")})

    def test_target_list(self):
        found = relations("Turla targets government, military and energy sectors.", ("Turla", "GROUP"),
                          ("government", "SECTOR"), ("military", "SECTOR"), ("energy", "SECTOR"))
        self.assertEqual(len(found), 3)

    def test_ground_truth_uses_and_targets_in_one_sentence(self):
        found = relations("TA415 used Cobalt Strike in a campaign against an entity in the airlines sector.",
                          ("TA415", "GROUP"), ("Cobalt Strike", "MALWARE"), ("airlines sector", "SECTOR"))
        self.assertIn(("USES", "TA415", "Cobalt Strike"), found)
        self.assertIn(("TARGETS", "TA415", "airlines sector"), found)


class SubjectAndDirectionErrors(unittest.TestCase):
    def test_tool_is_not_subject_of_group_campaign(self):
        found = relations("TA415 used Cobalt Strike in a campaign against an entity in the airlines sector.",
                          ("TA415", "GROUP"), ("Cobalt Strike", "MALWARE"), ("airlines sector", "SECTOR"))
        self.assertNotIn(("TARGETS", "Cobalt Strike", "airlines sector"), found)

    def test_delivered_through_is_passive(self):
        found = relations("Recently, we have also witnessed that Ryuk is delivered through the Bazar loader.",
                          ("Ryuk", "MALWARE"), ("Bazar", "MALWARE"))
        self.assertEqual(found, {("USES", "Bazar", "Ryuk")})

    def test_used_as_payload_is_not_active(self):
        found = relations("TinyNuke in 2017 used as a second-stage payload in a Zeus banking trojan campaign.",
                          ("TinyNuke", "MALWARE"), ("Zeus", "MALWARE"))
        self.assertNotIn(("USES", "TinyNuke", "Zeus"), found)

    def test_operators_are_the_subject_not_the_malware(self):
        self.assertEqual(relations("Trickbot operators now use BazarLoader.", ("Trickbot", "MALWARE"),
                                   ("BazarLoader", "MALWARE")), set())

    def test_new_human_subject_after_comma(self):
        found = relations("This group usually deploys CrimsonRAT, but this time they have deployed PeppyRAT.",
                          ("CrimsonRAT", "MALWARE"), ("PeppyRAT", "MALWARE"))
        self.assertNotIn(("USES", "CrimsonRAT", "PeppyRAT"), found)

    def test_clause_after_comma_keeps_group_as_subject(self):
        found = relations("APT-C-36 was seen using LimeRAT, targeting the same area through phishing emails.",
                          ("APT-C-36", "GROUP"), ("LimeRAT", "MALWARE"), ("phishing emails", "TECHNIQUE"))
        self.assertNotIn(("USES_TECHNIQUE", "LimeRAT", "phishing emails"), found)

    def test_agent_must_follow_by_directly(self):
        found = relations("Rebol script downloaded by MirrorBlast that drops ReflectiveGnome.",
                          ("Rebol", "MALWARE"), ("MirrorBlast", "MALWARE"), ("ReflectiveGnome", "MALWARE"))
        self.assertNotIn(("USES", "ReflectiveGnome", "Rebol"), found)


class NotAssertedRelations(unittest.TestCase):
    def test_comparison_is_not_use(self):
        self.assertEqual(relations("Ares uses the same function hashing algorithm as Kronos.",
                                   ("Ares", "MALWARE"), ("Kronos", "MALWARE")), set())

    def test_overlap_is_not_use(self):
        self.assertEqual(relations("TinyNuke infrastructure used in campaigns overlapped with PyLocky attacks.",
                                   ("TinyNuke", "MALWARE"), ("PyLocky", "MALWARE")), set())

    def test_other_malware_such_as_is_not_use(self):
        self.assertEqual(relations("REvil implemented a technique using Restart Manager also used by other "
                                   "ransomware such as SamSam.",
                                   ("REvil", "MALWARE"), ("SamSam", "MALWARE")), set())

    def test_impersonation_is_not_attribution(self):
        self.assertEqual(relations("A group posing as Lazarus Group threatened to hit Travelex with a DDoS attack.",
                                   ("Lazarus Group", "GROUP"), ("Travelex", "ORG")), set())

    def test_markdown_link_title_is_ignored(self):
        text = ('It is similar to Zeus, which has [many variants](/blog/zloader-loads-again '
                '"ZLoader Loads Again: New ZLoader Variant Returns") with identical functionality.')
        self.assertEqual(relations(text, ("Zeus", "MALWARE"), ("ZLoader", "MALWARE"), ("ZLoader", "MALWARE")), set())

    def test_hedged_relation_has_low_confidence(self):
        conf = confidences("Winnti was likely used to target Gravity.", ("Winnti", "GROUP"), ("Gravity", "ORG"))
        self.assertEqual(conf.get(("TARGETS", "Winnti", "Gravity")), "low")


class WrongObjectErrors(unittest.TestCase):
    def test_reporting_vendor_is_not_a_victim(self):
        self.assertEqual(relations("TA453 targets Proofpoint.", ("TA453", "GROUP"), ("Proofpoint", "ORG")), set())

    def test_source_of_reporting_is_not_a_victim(self):
        text = "Fancy Lazarus attacks are unconfirmed, FBI [reporting](https://x.example) indicates otherwise."
        self.assertEqual(relations(text, ("Fancy Lazarus", "GROUP"), ("FBI", "ORG")), set())

    def test_place_name_is_not_a_target(self):
        self.assertEqual(relations("Chinese Sea organizations were targeted by APT40.", ("Chinese", "LOC"),
                                   ("APT40", "GROUP")), set())

    def test_reason_clause_object_is_not_target(self):
        text = "The BlackSoul campaign targeted a government body, since NIST publishes standards."
        found = relations(text, ("BlackSoul", "MALWARE"), ("government body", "SECTOR"), ("NIST", "ORG"))
        self.assertNotIn(("TARGETS", "BlackSoul", "NIST"), found)

    def test_generic_malware_class_is_not_a_technique(self):
        self.assertEqual(relations("Maze was used in over 250 ransomware incidents.", ("Maze", "MALWARE"),
                                   ("ransomware", "TECHNIQUE")), set())


class FailuresFoundOnUnseenReports(unittest.TestCase):
    """Cases from the unseen-report audit (data/unseen_check_audit_provisional.csv)."""

    def test_negated_use_is_not_a_relation(self):
        self.assertEqual(relations("TA505 never used Emotet and only used TrickBot for a short period.",
                                   ("TA505", "GROUP"), ("Emotet", "MALWARE")), set())

    def test_negation_in_parenthesis_before_head(self):
        text = "unclear links between TA505 and Emotet (we've never seen TA505 use Emotet) as well as TrickBot."
        found = relations(text, ("TA505", "GROUP"), ("Emotet", "MALWARE"), ("TA505", "GROUP"),
                          ("Emotet", "MALWARE"), ("TrickBot", "MALWARE"))
        self.assertEqual({r for r in found if r[1] == "TA505" and r[2] == "TrickBot"}, set())

    def test_negation_far_away_does_not_block(self):
        self.assertIn(("USES", "TA551", "Qbot"),
                      relations("Proofpoint has observed TA551 distribute Qbot.", ("TA551", "GROUP"), ("Qbot", "MALWARE")))

    def test_product_name_is_not_a_victim(self):
        text = "TA577 compromised the victim via emails containing malicious Microsoft Office attachments."
        self.assertEqual(relations(text, ("TA577", "GROUP"), ("Microsoft", "ORG")), set())

    def test_vulnerable_product_is_not_a_victim(self):
        text = "Deployments of REvil were observed where attackers leveraged a vulnerability in Oracle WebLogic servers."
        self.assertEqual(relations(text, ("REvil", "MALWARE"), ("Oracle", "ORG")), set())

    def test_real_organization_victim_is_kept(self):
        self.assertEqual(relations("A group using Mount Locker attacked the Transportation Authority.",
                                   ("Mount Locker", "MALWARE"), ("Transportation Authority", "ORG")),
                         {("TARGETS", "Mount Locker", "Transportation Authority")})

    def test_malware_description_is_not_a_sector(self):
        text = "The final payload downloaded from these Upatre infections tends to be the Dyreza Banking Trojan."
        self.assertEqual(relations(text, ("Upatre", "MALWARE"), ("Banking", "SECTOR")), set())

    def test_technique_that_describes_the_malware_class(self):
        text = "NimzaLoader is a new initial access malware being distributed and used by the TA800 threat actor."
        self.assertEqual(relations(text, ("initial access", "TACTIC"), ("TA800", "GROUP")), set())

    def test_malware_cannot_use_itself(self):
        self.assertEqual(relations("TrickBot infections led to PowerBrace being dropped by TrickBot.",
                                   ("TrickBot", "MALWARE"), ("TrickBot", "MALWARE")), set())

    def test_generic_word_is_not_an_actor(self):
        self.assertEqual(relations("Chinese APT TA413 Resumes Targeting of Tibet.", ("APT", "GROUP"),
                                   ("Tibet", "LOC")), set())

    def test_used_in_reverses_direction(self):
        found = relations("Demystifying the Crypter Used in Emotet.", ("Crypter", "MALWARE"), ("Emotet", "MALWARE"))
        self.assertEqual(found, {("USES", "Emotet", "Crypter")})

    def test_reminiscent_campaign_is_not_use(self):
        text = "This campaign is highly reminiscent of the earlier one that utilized attachments to deliver ExileRAT."
        self.assertEqual(relations(text, ("campaign", "GROUP"), ("ExileRAT", "MALWARE")), set())

    def test_victims_research_field_is_not_a_victim_sector(self):
        text = "TA453 targeted individuals with a background in either genetics, oncology, or neurology."
        self.assertEqual(relations(text, ("TA453", "GROUP"), ("oncology", "SECTOR")), set())


class UsesFixesFromHumanReview(unittest.TestCase):
    """Cases from the human review of 66 unseen relations (data/unseen_check_audit_human_round2.csv)."""

    def test_same_operators_do_not_use_each_other(self):
        text = "Accounts used by the LastConn malware, the account used to deploy NimbleMamba, show that both " \
               "are deployed by the same operators."
        self.assertEqual(relations(text, ("LastConn", "MALWARE"), ("NimbleMamba", "MALWARE")), set())

    def test_replacement_is_not_use(self):
        text = "NimbleMamba is a replacement for the group's previously used LastConn implant."
        self.assertEqual(relations(text, ("NimbleMamba", "MALWARE"), ("LastConn", "MALWARE")), set())

    def test_noun_use_of_trigger_word(self):
        text = "Most of them were Zbot drops, Ransomware, Caphaw, Keylogger and more."
        self.assertEqual(relations(text, ("Zbot", "MALWARE"), ("Caphaw", "MALWARE")), set())

    def test_coordinated_clause_is_not_a_list_item(self):
        text = "We have seen RIG dropping Qakbot as well as Neutrino dropping CryptXXX."
        found = relations(text, ("RIG", "MALWARE"), ("Qakbot", "MALWARE"), ("Neutrino", "MALWARE"),
                          ("CryptXXX", "MALWARE"))
        self.assertIn(("USES", "RIG", "Qakbot"), found)
        self.assertNotIn(("USES", "RIG", "Neutrino"), found)

    def test_as_well_as_joining_two_objects_of_one_verb_keeps_both(self):
        found = relations("NimbleMamba uses the Dropbox API for both command and control as well as exfiltration.",
                          ("NimbleMamba", "MALWARE"), ("command and control", "TACTIC"), ("exfiltration", "TACTIC"))
        self.assertEqual({r[2] for r in found}, {"command and control", "exfiltration"})

    def test_both_objects_of_one_verb_in_a_list_are_kept(self):
        found = relations("TA505 campaigns use SDBot, Get2 and Dridex.", ("TA505", "GROUP"), ("SDBot", "TOOL"),
                          ("Get2", "MALWARE"), ("Dridex", "MALWARE"))
        self.assertEqual({r[2] for r in found}, {"SDBot", "Get2", "Dridex"})

    def test_unlike_marks_a_comparison(self):
        text = "Unlike njRAT it uses POST requests to exfiltrate data."
        self.assertEqual(relations(text, ("njRAT", "MALWARE"), ("exfiltrate", "TACTIC")), set())

    def test_relative_clause_belongs_to_the_intervening_entity(self):
        text = "TA429 used phishing macros that were later seen being used by LookBack."
        found = relations(text, ("TA429", "GROUP"), ("phishing macros", "TECHNIQUE"), ("LookBack", "MALWARE"))
        self.assertNotIn(("USES", "TA429", "LookBack"), found)

    def test_relative_clause_that_refers_to_the_head_is_kept(self):
        text = "We followed Squirrelwaffle that is being used to deliver Cobalt Strike."
        self.assertIn(("USES", "Squirrelwaffle", "Cobalt Strike"),
                      relations(text, ("Squirrelwaffle", "MALWARE"), ("Cobalt Strike", "MALWARE")))


def assertions(text, *mentions):
    """(relation, head, tail, polarity, certainty, confidence) for every assertion, positive or negative."""
    return {(rel, h["text"], t["text"], pol, cert, conf)
            for rel, h, t, _, _, conf, pol, cert in find_assertions(segment(text, *mentions), {})}


class AssertionContext(unittest.TestCase):
    """Polarity, certainty and scope: the context the contradiction-resolution stage needs."""

    def test_never_used_is_a_negative_assertion(self):
        found = assertions("TA505 never used Emotet and only used TrickBot for a short period.",
                           ("TA505", "GROUP"), ("Emotet", "MALWARE"))
        self.assertEqual({(r[0], r[1], r[2], r[3]) for r in found}, {("USES", "TA505", "Emotet", "negative")})

    def test_did_not_use_is_a_negative_assertion(self):
        found = assertions("TA505 did not use Emotet in this campaign.", ("TA505", "GROUP"), ("Emotet", "MALWARE"))
        self.assertEqual({r[3] for r in found}, {"negative"})

    def test_never_seen_x_use_y_is_a_negative_assertion(self):
        found = assertions("We have never seen TA505 use Emotet.", ("TA505", "GROUP"), ("Emotet", "MALWARE"))
        self.assertEqual({(r[0], r[1], r[2], r[3]) for r in found}, {("USES", "TA505", "Emotet", "negative")})

    def test_negative_assertions_are_not_plain_relations(self):
        seg = segment("TA505 never used Emotet.", ("TA505", "GROUP"), ("Emotet", "MALWARE"))
        self.assertEqual(find_relations(seg, {}), [])

    def test_distant_negation_is_not_a_negative_assertion(self):
        text = "We do not have visibility into the actual Fancy Lazarus DDoS attacks against banks."
        self.assertEqual(assertions(text, ("Fancy Lazarus", "GROUP"), ("banks", "SECTOR")), set())

    def test_positive_assertion_has_positive_polarity_and_asserted_certainty(self):
        found = assertions("APT29 frequently uses Cobalt Strike.", ("APT29", "GROUP"), ("Cobalt Strike", "MALWARE"))
        self.assertEqual({(r[3], r[4]) for r in found}, {("positive", "asserted")})

    def test_hedged_assertion_is_marked_hedged_with_low_confidence(self):
        found = assertions("Winnti was likely used to target Gravity.", ("Winnti", "GROUP"), ("Gravity", "ORG"))
        self.assertEqual({(r[3], r[4], r[5]) for r in found}, {("positive", "hedged", "low")})

    def test_scope_names_a_dubbed_campaign_and_dates(self):
        seg = {"clean_text": "In March 2020 TA413 ran a campaign dubbed Sepulcher against Europe.",
               "entities": [{"text": "March 2020", "type": "DATE"}]}
        scope = extract_scope(seg)
        self.assertEqual((scope["campaign"], scope["campaign_mentioned"], scope["dates"]),
                         ("Sepulcher", True, ["March 2020"]))

    def test_scope_takes_a_named_campaign_before_the_word(self):
        seg = {"clean_text": "The Squid Game campaign delivered Dridex.", "entities": []}
        self.assertEqual(extract_scope(seg)["campaign"], "Squid Game")

    def test_scope_generic_campaign_has_no_name(self):
        seg = {"clean_text": "A phishing campaign delivered Dridex.", "entities": []}
        scope = extract_scope(seg)
        self.assertEqual((scope["campaign"], scope["campaign_mentioned"]), (None, True))

    def test_scope_without_campaign_or_date(self):
        scope = extract_scope({"clean_text": "TA505 uses Dridex.", "entities": []})
        self.assertEqual((scope["campaign"], scope["campaign_mentioned"], scope["dates"]), (None, False, []))

    def test_date_spacing_is_cleaned(self):
        seg = {"clean_text": "On May 18 , 2021 TA505 ...", "entities": [{"text": "May 18 , 2021", "type": "DATE"}]}
        self.assertEqual(extract_scope(seg)["dates"], ["May 18, 2021"])


class DateParsing(unittest.TestCase):
    """Dates named in a sentence, turned into absolute ranges (event_time) where they can be pinned down."""

    PUB = "2021-06-10"

    def parse(self, text):
        return parse_date_mention(text, self.PUB)

    def span(self, text):
        r = self.parse(text)
        return (r["start"], r["end"], r["precision"]) if r else None

    def test_full_dates_in_several_formats(self):
        for text, day in (("Oct. 19, 2020", "2020-10-19"), ("September 29th, 2020", "2020-09-29"),
                          ("11 March 2021", "2021-03-11"), ("8 / 18 / 2021", "2021-08-18"),
                          ("2019 - 09 - 21", "2019-09-21"), ("2008 - May - 31", "2008-05-31")):
            self.assertEqual(self.span(text), (day, day, "day"), text)

    def test_day_first_numeric_date_when_the_first_number_cannot_be_a_month(self):
        self.assertEqual(self.span("13/05/2021"), ("2021-05-13", "2021-05-13", "day"))

    def test_month_and_year_covers_the_month(self):
        self.assertEqual(self.span("In April 2021"), ("2021-04-01", "2021-04-30", "month"))
        self.assertEqual(self.span("June \xa0 2021"), ("2021-06-01", "2021-06-30", "month"))

    def test_leap_year_february(self):
        self.assertEqual(self.span("February 2020"), ("2020-02-01", "2020-02-29", "month"))

    def test_year_covers_the_year_and_modifiers_do_not_narrow_it(self):
        self.assertEqual(self.span("2017"), ("2017-01-01", "2017-12-31", "year"))
        self.assertEqual(self.span("mid - 2020"), ("2020-01-01", "2020-12-31", "year"))

    def test_this_year_and_last_year_use_the_publication_date(self):
        self.assertEqual(self.span("this year"), ("2021-01-01", "2021-12-31", "year"))
        self.assertEqual(self.span("last year"), ("2020-01-01", "2020-12-31", "year"))
        self.assertEqual(self.parse("last year")["kind"], "relative")

    def test_relative_dates_without_a_publication_date_are_not_guessed(self):
        self.assertIsNone(parse_date_mention("last year", None))

    def test_month_without_a_year_is_not_guessed(self):
        self.assertIsNone(self.parse("In late June"))
        self.assertIsNone(self.parse("5 August"))

    def test_impossible_or_implausible_dates_are_rejected(self):
        for text in ("Feb 30, 2021", "1850", "recently", "today"):
            self.assertIsNone(self.parse(text), text)

    def test_event_time_is_the_span_of_the_dates_named(self):
        mentions = [self.parse("April 17, 2019"), self.parse("September 2020")]
        self.assertEqual(event_time(mentions), {"start": "2019-04-17", "end": "2020-09-30", "n_dates": 2})
        self.assertIsNone(event_time([]))

    def test_scope_collects_parsed_and_unresolved_dates(self):
        seg = {"clean_text": "In March 2020 and in late June TA413 ran a campaign.", "publication_date": self.PUB,
               "text": "In March 2020 and in late June TA413 ran a campaign.",
               "entities": [{"text": "March 2020", "type": "DATE", "char_start": 3},
                            {"text": "late June", "type": "DATE", "char_start": 20}]}
        scope = extract_scope(seg)
        self.assertEqual(scope["event_time"], {"start": "2020-03-01", "end": "2020-03-31", "n_dates": 1})
        self.assertEqual(scope["unresolved_dates"], ["late June"])
        self.assertEqual(scope["dates"], ["March 2020", "late June"])

    def test_since_makes_the_date_open_ended_up_to_the_report_date(self):
        text = "Active since at least 2007, APT28 primarily targets government organizations."
        seg = {"clean_text": text, "text": text, "publication_date": "2021-04-28",
               "entities": [{"text": "2007", "type": "DATE", "char_start": text.index("2007")}]}
        scope = extract_scope(seg)
        self.assertEqual(scope["event_time"]["start"], "2007-01-01")
        self.assertEqual(scope["event_time"]["end"], "2021-04-28")
        self.assertTrue(scope["date_mentions"][0]["open_ended"])

    def test_a_plain_year_is_not_open_ended(self):
        text = "In 2017 TA544 began a campaign."
        seg = {"clean_text": text, "text": text, "publication_date": "2021-04-28",
               "entities": [{"text": "2017", "type": "DATE", "char_start": 3}]}
        self.assertEqual(extract_scope(seg)["event_time"]["end"], "2017-12-31")


def triggers(text, *mentions):
    return {(rel, h["text"], t["text"]): trig for rel, h, t, trig, *_ in find_relations(segment(text, *mentions), {})}


class ImplicitTrigger(unittest.TestCase):
    """A technique entity that is itself a verb phrase needs no separate trigger word."""

    def test_actor_directly_followed_by_a_verb_phrase(self):
        found = triggers("Leviathan collects passwords to access other computers.", ("Leviathan", "GROUP"),
                         ("collects passwords", "TACTIC"))
        self.assertEqual(found, {("USES_TECHNIQUE", "Leviathan", "collects passwords"): "<implicit>"})

    def test_auxiliary_words_between_are_allowed(self):
        found = triggers("The prominent TA505 has returned to distributing large volumes of malicious emails.",
                         ("TA505", "GROUP"), ("distributing large volumes of malicious emails", "TECHNIQUE"))
        self.assertEqual(len(found), 1)

    def test_capability_wording(self):
        found = triggers("Cobalt Strike Beacon also has the capability to elevate privileges.",
                         ("Beacon", "MALWARE"), ("elevate privileges", "TACTIC"))
        self.assertEqual(len(found), 1)

    def test_a_bare_verb_is_not_enough(self):
        self.assertEqual(triggers("Once the address is located, Ares executes the module.", ("Ares", "MALWARE"),
                                  ("executes", "TECHNIQUE")), {})

    def test_a_noun_phrase_technique_without_a_trigger_is_not_a_relation(self):
        self.assertEqual(triggers("Leviathan credential dumping is common.", ("Leviathan", "GROUP"),
                                  ("credential dumping", "TACTIC")), {})

    def test_another_entity_between_blocks_it(self):
        text = "Beacon can elevate privileges and load and execute additional tools."
        found = triggers(text, ("Beacon", "MALWARE"), ("elevate privileges", "TACTIC"),
                         ("load and execute additional tools", "TECHNIQUE"))
        self.assertNotIn(("USES_TECHNIQUE", "Beacon", "load and execute additional tools"), found)

    def test_punctuation_between_blocks_it(self):
        self.assertEqual(triggers("Leviathan, collects passwords.", ("Leviathan", "GROUP"),
                                  ("collects passwords", "TACTIC")), {})

    def test_a_negated_implicit_relation_is_a_negative_assertion(self):
        found = assertions("Leviathan never collects passwords.", ("Leviathan", "GROUP"),
                           ("collects passwords", "TACTIC"))
        self.assertEqual({(r[0], r[3]) for r in found}, {("USES_TECHNIQUE", "negative")})

    def test_an_explicit_trigger_is_still_used_when_there_is_one(self):
        found = triggers("Leviathan uses collects passwords.", ("Leviathan", "GROUP"),
                         ("collects passwords", "TACTIC"))
        self.assertEqual(set(found.values()), {"uses"})


if __name__ == "__main__":
    unittest.main()
