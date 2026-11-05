-- Expand public.wellbeing_responses to the full Sri Lankan geography.
-- Existing rows are retained. The final VALIDATE step deliberately fails if
-- any existing row has a missing or invalid province/district pair.

DO $$
DECLARE
    geographic_check RECORD;
    check_definition TEXT;
BEGIN
    IF to_regclass('public.wellbeing_responses') IS NULL THEN
        RAISE EXCEPTION 'public.wellbeing_responses does not exist';
    END IF;

    FOR geographic_check IN
        SELECT conname, pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid = 'public.wellbeing_responses'::regclass
          AND contype = 'c'
    LOOP
        check_definition := geographic_check.definition;
        IF (
            check_definition ~* '\mprovince\M'
            AND check_definition ILIKE '%Western%'
        ) OR (
            check_definition ~* '\mdistrict\M'
            AND (
                check_definition ILIKE '%Colombo%'
                OR check_definition ILIKE '%Gampaha%'
                OR check_definition ILIKE '%Kalutara%'
            )
        ) THEN
            EXECUTE format(
                'ALTER TABLE public.wellbeing_responses DROP CONSTRAINT %I',
                geographic_check.conname
            );
        END IF;
    END LOOP;
END
$$;

ALTER TABLE public.wellbeing_responses
    DROP CONSTRAINT IF EXISTS wellbeing_responses_sri_lanka_geography_check;

ALTER TABLE public.wellbeing_responses
    ADD CONSTRAINT wellbeing_responses_sri_lanka_geography_check
    CHECK (
        province IS NOT NULL
        AND district IS NOT NULL
        AND (
            (province = 'Western' AND district IN ('Colombo', 'Gampaha', 'Kalutara'))
            OR (province = 'Central' AND district IN ('Kandy', 'Matale', 'Nuwara Eliya'))
            OR (province = 'Southern' AND district IN ('Galle', 'Matara', 'Hambantota'))
            OR (province = 'Northern' AND district IN ('Jaffna', 'Kilinochchi', 'Mannar', 'Mullaitivu', 'Vavuniya'))
            OR (province = 'Eastern' AND district IN ('Batticaloa', 'Ampara', 'Trincomalee'))
            OR (province = 'North Western' AND district IN ('Kurunegala', 'Puttalam'))
            OR (province = 'North Central' AND district IN ('Anuradhapura', 'Polonnaruwa'))
            OR (province = 'Uva' AND district IN ('Badulla', 'Monaragala'))
            OR (province = 'Sabaragamuwa' AND district IN ('Ratnapura', 'Kegalle'))
        )
    ) NOT VALID;

ALTER TABLE public.wellbeing_responses
    VALIDATE CONSTRAINT wellbeing_responses_sri_lanka_geography_check;
